from torch.utils.data import Dataset
from sklearn.metrics import classification_report
import pandas as pd
import lightning as L
import torch

LABELS = {'Fake':0, 'True':1}

class FakeSet(Dataset):
    def __init__(self, path):
        data = pd.read_json(path)
        self.text = list(data['text'].values)
        self.label = torch.tensor(data['category'].apply(lambda x: LABELS[x]),dtype=torch.long)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {'text':self.text[idx], 'label':self.label[idx]}

class FakeModel(torch.nn.Module):
    def __init__(self, model, output_size=2):
        super().__init__()
        self.model = model
        self.fc = torch.nn.Linear(model.config.hidden_size, output_size)

    def forward(self, x):
        x = torch.nn.functional.gelu(self.model(x).pooler_output)
        return self.fc(x)

class LightningModel(L.LightningModule):
    def __init__(self, model, tokenizer, opt, lr_scheduler = None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.opt = opt
        self.lr_scheduler = lr_scheduler   

    def forward(self, batch):
        batch = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)['input_ids'].to(self.device)
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        output = self.forward(batch['text'])
        loss = torch.nn.functional.cross_entropy(output, batch['label'].to(self.device))
        self.log('loss_train', loss)
        return loss

    def validation_step(self, batch, batch_idx):
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

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)