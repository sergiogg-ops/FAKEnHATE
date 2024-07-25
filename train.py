from transformers import RobertaModel, AutoTokenizer
from argparse import ArgumentParser
import lightning as L
import os
import utils
import torch

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

parser = ArgumentParser()
parser.add_argument('data_dir', help='Directory of the traning data')
parser.add_argument('save_dir', help='Directory to save the model')
parser.add_argument('-e','--epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('-b','--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('-lr','--learning_rate', type=float, default=5e-5, help='Learning rate')
parser.add_argument('-name','--experiment_name', type=str, default='baseline', help='Name of the experiment')
args = parser.parse_args()

utils.seed_everything(42)

tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
model = utils.FakeModel(RobertaModel.from_pretrained("PlanTL-GOB-ES/roberta-base-bne"))

train_set = utils.FakeSet('data/train.json')
dev_set = utils.FakeSet('data/dev.json')
test_set = utils.FakeSet('data/test.json')

num_workers = os.cpu_count() - 1
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

#optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, betas=(0.9,0.999), eps=1e-8)
lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.9, end_factor = 1, total_iters = args.epochs)

model = utils.LightningModel(model, tokenizer, optimizer, lr_scheduler)
save_dir = os.path.join(args.save_dir, args.experiment_name)
callbacks = [L.pytorch.callbacks.ModelCheckpoint(dirpath=save_dir, 
                                                 filename='{epoch}-{loss_val:.3f}-{f1_dev:.3f}', 
                                                 monitor='loss_dev', mode='min', save_top_k=3),
             L.pytorch.callbacks.EarlyStopping(monitor='loss_dev', mode='min', patience=3)]
#logger = L.pytorch.loggers.CSVLogger('logs')
logger = L.pytorch.loggers.MLFlowLogger(save_dir='logs', experiment_name=args.experiment_name)
trainer = L.Trainer(max_epochs=args.epochs,
                    logger=logger,
                    callbacks=callbacks)
trainer.fit(model, train_loader, dev_loader)
#trainer.test(model, test_loader)