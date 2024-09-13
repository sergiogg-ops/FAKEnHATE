from torch.utils.data import Dataset
from sklearn.metrics import classification_report
from einops import rearrange
from tqdm import tqdm
from math import ceil
import pandas as pd
import lightning as L
import numpy as np
import torch
import spacy

LABELS = {'Fake':0, 'True':1}
ID2LABEL = {0:'Fake', 1:'True'}

class FakeSet(Dataset):
    '''
    Dataset for fake news classification
    '''
    def __init__(self, path, sep = '', masker = None, verbose = False):
        '''
        Parameters:
            path: str, path to the json file that contains the data
            sep: str, separator to add between the headline and the text
            masker: function, function to mask named entities
            verbose: bool, show progress bar
        '''
        data = pd.read_json(path)
        self.text = list(data['text'].values)
        self.label = torch.tensor(data['category'].apply(lambda x: LABELS[x]),dtype=torch.long)
        self.topic = list(data['topic'])
        if 'headline' in data.columns:
            self.text = [h + sep + t if h else t for h,t in zip(data['headline'], self.text)]
        if masker:
            self.text = [masker(t) for t in tqdm(self.text, desc='Enmascarando entidades', disable=not verbose, unit='noticia')]

    def __len__(self):
        '''
        Returns the number of samples in the dataset
        '''
        return len(self.label)

    def __getitem__(self, idx):
        '''
        Returns a sample from the dataset

        Parameters:
            idx: int, index of the sample
        
        Returns:
            dict, {'text':str, 'label':int}
        '''
        return {'text':self.text[idx], 'label':self.label[idx]}

class FakeModel(torch.nn.Module):
    '''
    Model for fake news classification
    '''
    def __init__(self, model, tokenizer, output_size=2, add_noise=False):
        '''
        Parameters:
            model: transformers model, pre-trained BERT like model
            output_size: int, number of classes
            add_noise: str or bool, type of noise for the embeddings
                - False: no noise will be applied
                - uniform: uniform noise
                - normal: gaussian noise
        '''
        super().__init__()
        self.extractor = model
        self.tokenizer = tokenizer
        self.head = torch.nn.Sequential(
            torch.nn.GELU(),
            #torch.nn.Dropout(0.3),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(model.config.hidden_size, output_size))
        self.add_noise = bool(add_noise)
        self.noise = {'uniform':rand_like,
                    'normal':torch.randn_like,
                    False:None}[add_noise]

    def forward(self, batch, alpha = 1, output_attentions = False):
        '''
        Make an inference with the model

        Parameters:
            x: torch.Tensor, input tensor
            attn_mask: torch.Tensor, attention mask
            alpha: int, scaling factor for the noisy embeddings
        '''
        batch = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=self.tokenizer.model_max_length)
        x, attn_mask = batch['input_ids'].to(self.extractor.device), batch['attention_mask'].to(self.extractor.device)
        if self.add_noise and self.noise:
            embs = self.extractor.embeddings(x, past_key_values_length=0)
            scale = alpha / torch.unsqueeze(torch.sqrt(torch.sum(attn_mask, dim=1,keepdim=True) * embs.size(-1)),2).expand(-1,x.shape[-1],embs.size(-1))
            noise = self.noise(embs) * scale
            embs = embs + noise * torch.unsqueeze(attn_mask,2).expand(-1,-1,embs.size(-1))
            model_output = self.extractor(inputs_embeds=embs, attention_mask=attn_mask, output_attentions=output_attentions)
        else:
            model_output = self.extractor(x, attn_mask, output_attentions=output_attentions)
        if output_attentions:
            return self.head(model_output.pooler_output), model_output.attentions
        return self.head(model_output.pooler_output)
    
    def train(self, mode = True):
        '''
        Set the model in training mode

        Parameters:
            mode: bool, training mode
        
        Returns:
            FakeModel, model in training mode
        '''
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.add_noise = mode
        return self

class FakeBELT(FakeModel):
    '''
    Model for fake news classification with longer context
    '''
    def __init__(self, model, tokenizer, output_size=2, add_noise=False, pool = 'max', step = 0.75, max_length = 100000):
        '''
        Parameters:
            model: transformers model, pre-trained BERT like model
            output_size: int, number of classes
            add_noise: str or bool, type of noise for the embeddings
                - False: no noise will be applied
                - uniform: uniform noise
                - normal: gaussian noise
        '''
        super().__init__(model,tokenizer,output_size,add_noise)
        self.pool_strategy = pool
        if pool == 'rnn':
            self.rnn = torch.nn.RNN(model.config.hidden_size,model.config.hidden_size,batch_first=True)
        elif pool == 'lstm':
            self.lstm = torch.nn.LSTM(model.config.hidden_size,model.config.hidden_size,batch_first=True, dropout=0.3)
        elif pool == 'transf':
            self.overtransformer = torch.nn.TransformerEncoderLayer(model.config.hidden_size,
                                                                    8, batch_first=True,
                                                                    norm_first=True)
            #self.overtransformer = CustomEncoder(1)
        self.extractor.pooler = torch.nn.Sequential(
            torch.nn.LayerNorm(model.config.hidden_size),
            self.extractor.pooler.dense,
            self.extractor.pooler.activation
        )
        self.step = step
        self.cls = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        self.eos = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        self.pad = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.max_length = max_length

    def forward(self, batch, alpha = 1):
        '''
        Make an inference with the model

        Parameters:
            batch: list of str, input texts
            alpha: int, scaling factor for the noisy embeddings
        '''
        CHUNK_SIZE = self.tokenizer.model_max_length - 2
        STEP = int(CHUNK_SIZE * self.step)
        # TOKENIZAR
        batch = self.tokenizer(batch, return_tensors='pt', padding=True, max_length=self.max_length, truncation=True)
        x, attn_mask = batch['input_ids'].to(self.extractor.device), batch['attention_mask'].to(self.extractor.device)
        x, attn_mask = x[:,1:], attn_mask[:,1:] # eliminar cls
        max_len = x.shape[-1]
        lengths = torch.sum(attn_mask, dim=1)
        # PARTIR Y CHUNKS
        num_chunks = max(ceil((max_len - CHUNK_SIZE) / STEP),0)
        padding = num_chunks * STEP + CHUNK_SIZE - max_len
        x = torch.cat([x, torch.ones(x.shape[0],padding, dtype=torch.long).to(x.device) * self.pad], dim=1)
        x = rearrange(x.unfold(1,CHUNK_SIZE,STEP), 'b n l -> (b n) l')
        attn_mask = torch.cat([attn_mask, torch.ones(attn_mask.shape[0],padding,dtype=torch.long).to(attn_mask.device)], dim=1)
        attn_mask = rearrange(attn_mask.unfold(1,CHUNK_SIZE,STEP), 'b n l -> (b n) l')
        # CLS Y EOS
        x = torch.cat([torch.ones(x.shape[0],1,dtype=torch.long).to(x.device) * self.cls, x, torch.ones(x.shape[0],1,dtype=torch.long).to(x.device) * self.eos], dim=1)
        attn_mask = torch.cat([torch.ones(attn_mask.shape[0],1,dtype=torch.long).to(attn_mask.device), attn_mask, torch.ones(x.shape[0],1,dtype=torch.long).to(x.device)], dim=1)
        i = torch.tensor([torch.sum(torch.ceil(lengths[:j+1]/CHUNK_SIZE),dtype=torch.int) for j in range(lengths.shape[0])])
        x[i-1,-1] = self.pad
        attn_mask[i-1,-1] = 0
        # EMBEDDINGS
        embs = self.extractor.embeddings(x)
        with torch.no_grad():
            if self.add_noise and self.noise:
                scale = alpha / torch.unsqueeze(torch.sqrt(torch.sum(attn_mask, dim=1,keepdim=True) * embs.size(-1)),2).expand(-1,x.shape[-1],embs.size(-1))
                noise = self.noise(embs) * scale
                embs = embs + noise * torch.unsqueeze(attn_mask,2).expand(-1,-1,embs.size(-1))
        # FORWARD
        hidden_state = self.extractor(inputs_embeds=embs, attention_mask=attn_mask).last_hidden_state
        hidden_state *= torch.unsqueeze(attn_mask,-1).expand(-1,-1,hidden_state.size(-1))
        # POOLING
        hidden_state = rearrange(hidden_state[:,0,:], '(b n) h -> b n h', n = num_chunks + 1)
        hidden_state = self.pool(hidden_state, dim=1)
        model_output = self.extractor.pooler(hidden_state)
        # CLASIFICACION
        return self.head(model_output)

    def pool(self, x, dim = 1):
        '''
        Performs the pooling operation over the hidden states

        Parameters:
            x: torch.Tensor, hidden states
            dim: int, dimension to pool
        
        Returns:
            torch.Tensor, pooled hidden states
        '''
        if self.pool_strategy == 'max':
            return torch.max(x,dim=dim).values
        elif self.pool_strategy == 'avg':
            return torch.mean(x,dim=dim)
        elif self.pool_strategy == 'sum':
            return torch.sum(x,dim=dim)
        elif self.pool_strategy == 'attn':
            hidden_size = x.shape[-1]
            x = rearrange(x, 'b n h -> b (n h)')
            x = torch.nn.functional.softmax(x @ x.transpose(0,1) / torch.sqrt(torch.tensor(x.shape[1])), dim=1) @ x
            return x[:,:hidden_size]
        elif self.pool_strategy == 'rnn':
            return self.rnn(x,torch.unsqueeze(torch.ones(x.shape[0],x.shape[-1]),0).to(x.device))[1][0]
        elif self.pool_strategy == 'lstm':
            return self.lstm(x)[0][:,-1]
        elif self.pool_strategy == 'transf':
            return self.overtransformer(x)[:,0]
        else:
            raise ValueError(f'Unknown pooling strategy: {self.pool_strategy}')
    
class CustomEncoder(torch.nn.Module):
    '''
    Custom transformer encoder with the attention layer before the feed forward layer
    '''
    def __init__(self, num_layers):
        '''
        Parameters:
            num_layers: int, number of transformer layers
        '''
        super().__init__()
        self.mlps = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.GELU(),
            torch.nn.Linear(768, 768),
            torch.nn.GELU(),
        ) for _ in range(num_layers)])
        self.attns = torch.nn.ModuleList([torch.nn.MultiheadAttention(768, 8, batch_first=True) for _ in range(num_layers)])
        self.norms1 = torch.nn.ModuleList([torch.nn.LayerNorm(768) for _ in range(num_layers)])
        self.norms2 = torch.nn.ModuleList([torch.nn.LayerNorm(768) for _ in range(num_layers)])
    
    def forward(self, x):
        '''
        Parameters:
            x: torch.Tensor, input tensor
        '''
        for mlp, attn, norm1, norm2 in zip(self.mlps, self.attns, self.norms1, self.norms2):
            x = x + attn(x,x,x)[0]
            x = x + mlp(norm1(x))
            x = norm2(x)
        return x

class CustomBELT(FakeBELT):
    '''
    Modificiation of the original BELT that performs the aggregation along the hidden states instead of the CLS tokens
    '''
    def __init__(self, model, tokenizer, output_size=2, add_noise=False, step = 0.75, max_length = 100000):
        '''
        Parameters:
            model: transformers model, pre-trained BERT like model
            output_size: int, number of classes
            add_noise: str or bool, type of noise for the embeddings
                - False: no noise will be applied
                - uniform: uniform noise
                - normal: gaussian noise
            step: float, step for the chunking process
            max_length: int, max length of the text
        '''
        super().__init__(model,tokenizer,output_size,add_noise)
        self.aggregator = torch.nn.MultiheadAttention(model.config.hidden_size, 8, batch_first=True, kdim=model.config.hidden_size, vdim=model.config.hidden_size)
        self.sticky = torch.nn.Parameter(torch.randn(1,tokenizer.model_max_length,model.config.hidden_size))
        self.top_encoder = torch.nn.TransformerEncoderLayer(model.config.hidden_size,
                                                                    8, batch_first=True,
                                                                    norm_first=True)
        #self.lstm = torch.nn.LSTM(model.config.hidden_size,model.config.hidden_size,batch_first=True, dropout=0.3)

        self.step = step
        self.cls = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        self.eos = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        self.pad = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.max_length = max_length
    
    def forward(self, batch, alpha = 1):
        '''
        Parameters:
            batch: list of str, input texts
            alpha: int, scaling factor for the noisy embeddings
        '''
        CHUNK_SIZE = self.tokenizer.model_max_length - 2
        STEP = int(CHUNK_SIZE * self.step)
        # TOKENIZAR
        batch = self.tokenizer(batch, return_tensors='pt', padding=True, max_length=self.max_length, truncation=True)
        x, attn_mask = batch['input_ids'].to(self.extractor.device), batch['attention_mask'].to(self.extractor.device)
        BATCH = x.shape[0]
        x, attn_mask = x[:,1:], attn_mask[:,1:] # eliminar cls
        max_len = x.shape[-1]
        lengths = torch.sum(attn_mask, dim=1)
        # PARTIR Y CHUNKS
        num_chunks = max(ceil((max_len - CHUNK_SIZE) / STEP),0)
        padding = num_chunks * STEP + CHUNK_SIZE - max_len
        x = torch.cat([x, torch.zeros(x.shape[0],padding, dtype=torch.long).to(x.device) + self.pad], dim=1)
        x = rearrange(x.unfold(1,CHUNK_SIZE,STEP), 'b n l -> (b n) l')
        # TODO: ZEROS POR ONES NO?????
        attn_mask = torch.cat([attn_mask, torch.zeros(attn_mask.shape[0],padding,dtype=torch.long).to(attn_mask.device)], dim=1)
        attn_mask = rearrange(attn_mask.unfold(1,CHUNK_SIZE,STEP), 'b n l -> (b n) l')
        # CLS Y EOS
        x = torch.cat([torch.zeros(x.shape[0],1,dtype=torch.long).to(x.device) + self.cls, x, torch.zeros(x.shape[0],1,dtype=torch.long).to(x.device) + self.eos], dim=1)
        attn_mask = torch.cat([torch.ones(attn_mask.shape[0],1,dtype=torch.long).to(attn_mask.device), attn_mask, torch.ones(x.shape[0],1,dtype=torch.long).to(x.device)], dim=1)
        i = torch.tensor([torch.sum(torch.ceil(lengths[:j+1]/CHUNK_SIZE),dtype=torch.int) for j in range(lengths.shape[0])])
        x[i-1,-1] = self.pad
        attn_mask[i-1,-1] = 0
        # EMBEDDINGS
        embs = self.extractor.embeddings(x)
        with torch.no_grad():
            if self.add_noise and self.noise:
                scale = alpha / torch.unsqueeze(torch.sqrt(torch.sum(attn_mask, dim=1,keepdim=True) * embs.size(-1)),2).expand(-1,x.shape[-1],embs.size(-1))
                noise = self.noise(embs) * scale
                embs = embs + noise * torch.unsqueeze(attn_mask,2).expand(-1,-1,embs.size(-1))
        # 1º FASE [b n, l, h]
        hidden_state = self.extractor(inputs_embeds=embs, attention_mask=attn_mask).last_hidden_state
        hidden_state *= torch.unsqueeze(attn_mask,-1).expand(-1,-1,hidden_state.size(-1))
        # AGREGACION [b l, n, h]
        hidden_state = rearrange(hidden_state, '(b n) l h -> (b l) n h', n = num_chunks + 1)
        sticker = self.sticky.expand(BATCH,-1,-1)
        sticker = torch.unsqueeze(rearrange(sticker, 'b l h -> (b l) h'),1)
        hidden_state = torch.cat([sticker, hidden_state], dim=1)
        hidden_state = self.aggregator(hidden_state, hidden_state, hidden_state, need_weights=False)[0][:,0]
        # 2º FASE [b, l, h]
        hidden_state = rearrange(hidden_state, '(b l) h -> b l h', b=BATCH)
        hidden_state = self.top_encoder(hidden_state)[:,0]
        # hidden_state = rearrange(hidden_state[:,0,:], '(b n) h -> b n h', n = num_chunks + 1)
        # hidden_state = self.lstm(hidden_state)[0][:,-1]
        # CLASIFICACION [b, h]
        return self.head(hidden_state)
        
class LightningModel(L.LightningModule):
    '''
    Envelope lightning module for fake news classification
    '''
    def __init__(self, model, tokenizer, lr = 5e-6, sch_factor = 0.7, sch_iters = 10, alpha = 1, unfreeze = 0):
        '''
        Parameters:
            model: FakeModel, model for fake news classification
            tokenizer: transformers tokenizer, tokenizer for the model
            lr: float, learning rate
            sch_factor: start factor for the linear learning rate scheduler
            sch_iters: number of iteration for the linear learning rate scheduler
            alpha: scaling factor of the noisy embeddings to apply overt the model
        '''
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.sch_factor = sch_factor
        self.sch_iters = sch_iters
        self.alpha = alpha
        self.unfreeze_epoch = unfreeze
        if unfreeze >= 0:
            for p in self.model.extractor.encoder.parameters(): p.requires_grad = False
        if hasattr(self.model, 'step'):
            self.hparams.step = self.model.step
        if hasattr(self.model, 'max_length'):
            self.hparams.max_length = self.model.max_length
        if hasattr(self.model, 'pool_strategy'):
            self.hparams.pool = self.model.pool_strategy
        self.save_hyperparameters(ignore=['model','tokenizer'])

    def forward(self, batch):
        '''
        Common forward step of the model

        Parameters:
            batch: str, batch of texts
        
        Returns:
            torch.Tensor, output of the model
        '''
        #batch = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        #return self.model(batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device), alpha = self.alpha)
        return self.model(batch, alpha = self.alpha)

    def training_step(self, batch, batch_idx):
        '''
        Batch processing for training

        Parameters:
            batch: dict, {'text':str, 'label':int}
            batch_idx: int, index of the batch
        
        Returns:
            torch.Tensor, loss
        '''
        self.model.train()
        output = self.forward(batch['text'])
        loss = torch.nn.functional.cross_entropy(output, batch['label'].to(self.device))
        self.log('loss_train', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        '''
        Batch processing for validation. Loads the metrics to the logger

        Parameters:
            batch: dict, {'text':str, 'label':int}
            batch_idx: int, index of the batch
        
        Returns:
            dict, classification report
        '''
        loss, report = self.shared_step(batch, batch_idx)
        metrics = {}
        metrics['loss_dev'] = loss
        metrics['accuracy_dev'] = report['accuracy']
        metrics['f1_dev'] = report['macro avg']['f1-score']
        metrics['f1_true_dev'] = report['True']['f1-score']
        metrics['f1_fake_dev'] = report['Fake']['f1-score']
        metrics['recall_true_dev'] = report['True']['recall']
        metrics['recall_fake_dev'] = report['Fake']['recall']
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        '''
        Batch processing for test. Loads the metrics to the logger
        
        Parameters:
            batch: dict, {'text':str, 'label':int}
            batch_idx: int, index of the batch
        
        Returns:
            dict, classification report
        '''
        loss, report = self.shared_step(batch, batch_idx)
        metrics = {}
        metrics['loss_test'] = loss
        metrics['accuracy_test'] = report['accuracy']
        metrics['f1_test'] = report['macro avg']['f1-score']
        metrics['f1_true_test'] = report['True']['f1-score']
        metrics['f1_fake_test'] = report['Fake']['f1-score']
        metrics['recall_true_test'] = report['True']['recall']
        metrics['recall_fake_test'] = report['Fake']['recall']
        self.log_dict(metrics)
        return metrics

    def shared_step(self,batch, batch_idx):
        '''
        Shared step for validation and test

        Parameters:
            batch: dict, {'text':str, 'label':int}
            batch_idx: int, index of the batch
        
        Returns:
            torch.Tensor, loss
            dict, classification report
        '''
        self.model.eval()
        output = self.forward(batch['text'])
        loss = torch.nn.functional.cross_entropy(output, batch['label'].to(self.device))
        report = classification_report(batch['label'].to('cpu'), 
                                       output.argmax(dim=1).to('cpu'), 
                                       target_names=['Fake','True'], 
                                       output_dict=True,
                                       zero_division=0,
                                       labels=[0,1])
        return loss, report

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch['text'])#.argmax(dim=1)

    def configure_optimizers(self):
        out = {'optimizer': torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9,0.999), eps=1e-8)}
        out['lr_scheduler'] = {'scheduler': torch.optim.lr_scheduler.LinearLR(out['optimizer'], 
                                                                start_factor = self.sch_factor, 
                                                                end_factor = 1, 
                                                                total_iters = self.sch_iters),
                                    'monitor': 'f1_dev',
                                    #'interval': 'epoch',
                                    'interval': 'step',
                                    'frequency': 124}
        return out
    
    def on_train_epoch_start(self):
        '''
        Unfreezes the model parameters if the current epoch has been set as the unfreeze epoch
        '''
        if self.current_epoch == self.unfreeze_epoch and hasattr(self.model.extractor, 'encoder'):
            for p in self.model.extractor.encoder.parameters(): p.requires_grad = True

class NamedEntityMasker:
    '''
    Envelope for masking named entities with spacy module
    '''
    def __init__(self,ents = []):
        '''
        Parameters:
            ents: list, list of entities to mask, if emtpy all entities are masked
        '''
        self.model = spacy.load('es_core_news_sm')
        if ents == []:
            ents = self.model.get_pipe('ner').labels
        else:
            self.ents = ents
    
    def __call__(self, text):
        '''
        Mask named entities in the text

        Parameters:
            text: str, text to mask
        
        Returns:
            str, text with entities masked
        '''
        doc = self.model(text)
        for ent in doc.ents:
            if ent.label_ in self.ents:
                text = text.replace(ent.text, f'*{ent.label_}*')
        return text

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    L.seed_everything(seed)

def rand_like(x):
    '''
    Returns a tensor with the same shape as x with random values distributed uniformally between -1 and 1

    Parameters:
        x: torch.Tensor, tensor to get the shape
    
    Returns:
        torch.Tensor, tensor with random values
    '''
    return torch.rand_like(x) * 2 - 1