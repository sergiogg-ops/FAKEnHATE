from torch.utils.data import Dataset
from sklearn.metrics import classification_report
from tqdm import tqdm
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
    def __init__(self, model, output_size=2):
        '''
        Parameters:
            model: transformers model, pre-trained model
            output_size: int, number of classes
        '''
        super().__init__()
        self.model = model
        self.fc = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(model.config.hidden_size, output_size))

    def forward(self, x, attn_mask):
        '''
        Make an inference with the model

        Parameters:
            x: torch.Tensor, input tensor
            attn_mask: torch.Tensor, attention mask
        '''
        return self.fc(self.model(x, attn_mask).pooler_output)

class LightningModel(L.LightningModule):
    '''
    Lightning module envelope for fake news classification
    '''
    def __init__(self, model, tokenizer, opt, lr_scheduler = None):
        '''
        Parameters:
            model: FakeModel, model for fake news classification
            tokenizer: transformers tokenizer, tokenizer for the model
            opt: torch.optim.Optimizer, optimizer
            lr_scheduler: torch.optim.lr_scheduler._LRScheduler, learning rate scheduler
        '''
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.opt = opt
        self.lr_scheduler = lr_scheduler   

    def forward(self, batch):
        '''
        Common forward step of the model

        Parameters:
            batch: str, batch of texts
        
        Returns:
            torch.Tensor, output of the model
        '''
        batch = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        return self.model(batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device))

    def training_step(self, batch, batch_idx):
        '''
        Batch processing for training

        Parameters:
            batch: dict, {'text':str, 'label':int}
            batch_idx: int, index of the batch
        
        Returns:
            torch.Tensor, loss
        '''
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
        output = self.forward(batch['text'])
        loss = torch.nn.functional.cross_entropy(output, batch['label'].to(self.device))
        report = classification_report(batch['label'].to('cpu'), 
                                       output.argmax(dim=1).to('cpu'), 
                                       target_names=['Fake','True'], 
                                       output_dict=True,
                                       zero_division=0)
        return loss, report

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch['text']).argmax(dim=1)

    def configure_optimizers(self):
        return {'optimizer': self.opt,
                'lr_scheduler': self.lr_scheduler}

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