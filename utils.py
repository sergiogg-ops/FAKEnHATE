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
        self.model = model
        self.tokenizer = tokenizer
        self.head = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(model.config.hidden_size, output_size))
        self.add_noise = bool(add_noise)
        self.noise = {'uniform':rand_like,
                    'normal':torch.randn_like,
                    False:None}[add_noise]

    def forward(self, batch, alpha = 1):
        '''
        Make an inference with the model

        Parameters:
            x: torch.Tensor, input tensor
            attn_mask: torch.Tensor, attention mask
            alpha: int, scaling factor for the noisy embeddings
        '''
        batch = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=self.tokenizer.model_max_length)
        x, attn_mask = batch['input_ids'].to(self.model.device), batch['attention_mask'].to(self.model.device)
        if self.add_noise and self.noise:
            embs = self.model.embeddings(x, past_key_values_length=0)
            scale = alpha / torch.unsqueeze(torch.sqrt(torch.sum(attn_mask, dim=1,keepdim=True) * embs.size(-1)),2).expand(-1,x.shape[-1],embs.size(-1))
            noise = self.noise(embs) * scale
            embs = embs + noise * torch.unsqueeze(attn_mask,2).expand(-1,-1,embs.size(-1))
            model_output = self.model(inputs_embeds=embs, attention_mask=attn_mask)
        else:
            model_output = self.model(x, attn_mask)
        return self.head(model_output.pooler_output)
    
    def train(self, mode = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.add_noise = mode
        return self

class FakeBELT(FakeModel):
    def __init__(self, model, tokenizer, output_size=2, add_noise=False, pool = 'max', step = 0.75):
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
        self.model.pooler = torch.nn.Sequential(
            torch.nn.Linear(model.config.hidden_size, model.config.hidden_size),
            torch.nn.Tanh(),
        )
        self.pool_strategy = pool
        self.step = step
        self.cls = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        self.eos = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        self.pad = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    '''
    Model for fake news classification using all the text
    '''
    def forward(self, batch, alpha = 1):
        '''
        Make an inference with the model

        Parameters:
            x: torch.Tensor, input tensor
            attn_mask: torch.Tensor, attention mask
            alpha: int, scaling factor for the noisy embeddings
        '''
        CHUNK_SIZE = self.tokenizer.model_max_length - 1
        STEP = int(CHUNK_SIZE * self.step)
        # TOKENIZAR
        batch = self.tokenizer(batch, return_tensors='pt', padding=True)
        x, attn_mask = batch['input_ids'].to(self.model.device), batch['attention_mask'].to(self.model.device)
        x, attn_mask = x[:,1:], attn_mask[:,1:] # eliminar cls
        max_len = x.shape[-1]
        lengths = torch.sum(attn_mask, dim=1)
        # PARTIR Y CHUNKS
        num_chunks = ceil((max_len - CHUNK_SIZE) / STEP)
        padding = num_chunks * STEP + CHUNK_SIZE - max_len
        x = torch.cat([x, torch.ones(x.shape[0],padding, dtype=torch.long).to(x.device) * self.pad], dim=1)
        x = rearrange(x.unfold(1,CHUNK_SIZE,STEP), 'b n l -> (b n) l')
        attn_mask = torch.cat([attn_mask, torch.ones(attn_mask.shape[0],padding,dtype=torch.long).to(attn_mask.device)], dim=1)
        attn_mask = rearrange(attn_mask.unfold(1,CHUNK_SIZE,STEP), 'b n l -> (b n) l')
        # CLS Y EOS
        x = torch.cat([torch.ones(x.shape[0],1,dtype=torch.long).to(x.device) * self.cls, x], dim=1)
        attn_mask = torch.cat([torch.ones(attn_mask.shape[0],1,dtype=torch.long).to(attn_mask.device), attn_mask], dim=1)
        x[:,lengths%CHUNK_SIZE+1] = self.eos
        attn_mask[:,lengths%CHUNK_SIZE+1] = 1
        # EMBEDDINGS
        embs = self.model.embeddings(x)
        if self.add_noise and self.noise:
            scale = alpha / torch.unsqueeze(torch.sqrt(torch.sum(attn_mask, dim=1,keepdim=True) * embs.size(-1)),2).expand(-1,x.shape[-1],embs.size(-1))
            noise = self.noise(embs) * scale
            embs = embs + noise * torch.unsqueeze(attn_mask,2).expand(-1,-1,embs.size(-1))
        # FORWARD
        hidden_state = self.model(x, attention_mask=attn_mask).last_hidden_state
        hidden_state *= torch.unsqueeze(attn_mask,-1).expand(-1,-1,hidden_state.size(-1))
        # POOLING
        hidden_state = rearrange(hidden_state[:,0,:], '(b n) h -> b n h', n = num_chunks + 1)
        hidden_state = self.pool(hidden_state, lengths, dim=1)
        model_output = self.model.pooler(hidden_state)
        # CLASIFICACION
        return self.head(model_output)

    def pool(self, x, lengths, dim = 1):
        if self.pool_strategy == 'max':
            return torch.max(x,dim=dim).values
        elif self.pool_strategy == 'mean':
            return torch.sum(x,dim=dim) / lengths
        elif self.pool_strategy == 'sum':
            return torch.sum(x,dim=dim)
        else:
            raise ValueError(f'Unknown pooling strategy: {self.pool_strategy}')
        
class LightningModel(L.LightningModule):
    '''
    Envelope lightning module for fake news classification
    '''
    def __init__(self, model, tokenizer, lr = 5e-6, sch_start_factor = 0.7, sch_iters = 10, alpha = 1):
        '''
        Parameters:
            model: FakeModel, model for fake news classification
            tokenizer: transformers tokenizer, tokenizer for the model
            lr: float, learning rate
            sch_start_factor: start factor for the linear learning rate scheduler
            sch_iters: number of iteration for the linear learning rate scheduler
            alpha: scaling factor of the noisy embeddings to apply overt the model
        '''
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.sch_start_factor = sch_start_factor
        self.sch_iters = sch_iters
        self.alpha = alpha
        self.save_hyperparameters('lr', 'sch_start_factor', 'sch_iters', 'alpha')  

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
        self.log_dict(metrics)
        return report

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
        #output = output.unsqueeze(0) if len(output.shape) == 1 else output
        report = classification_report(batch['label'].to('cpu'), 
                                       output.argmax(dim=1).to('cpu'), 
                                       target_names=['Fake','True'], 
                                       output_dict=True,
                                       zero_division=0,
                                       labels=[0,1])
        return loss, report

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch['text']).argmax(dim=1)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9,0.999), eps=1e-8)
        return {'optimizer': opt,
                'lr_scheduler': {'scheduler': torch.optim.lr_scheduler.LinearLR(opt, 
                                                                  start_factor = self.sch_start_factor, 
                                                                  end_factor = 1, 
                                                                  total_iters = self.sch_iters),
                                        'monitor': 'f1_dev',
                                        'interval': 'epoch',
                                        'frequency': 1}}

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
    return torch.rand_like(x) * 2 - 1