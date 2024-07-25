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
parser.add_argument('-lr','--learning_rate', type=float, default=2e-5, help='Learning rate')
parser.add_argument('-run','--run_name', type=str, default='run', help='Name of the run')
parser.add_argument('-ner','--mask_ner', action='store_true', help='Mask named entities')
parser.add_argument('-lr_sch','--lr_scheduler', type=float, help='Start factor for the linear scheduler')
args = parser.parse_args()

experiment_name = ''
if args.mask_ner:
    experiment_name += 'ner_'
if experiment_name == '':
    experiment_name = 'baseline'

utils.seed_everything(42)

tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
model = utils.FakeModel(RobertaModel.from_pretrained("PlanTL-GOB-ES/roberta-base-bne"))

train_set = utils.FakeSet('data/train.json', sep=tokenizer.sep_token)
dev_set = utils.FakeSet('data/dev.json', sep=tokenizer.sep_token)
test_set = utils.FakeSet('data/test.json', sep=tokenizer.sep_token)

num_workers = os.cpu_count() - 1
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9,0.999), eps=1e-8)
lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = args.lr_scheduler, end_factor = 1, total_iters = args.epochs)

model = utils.LightningModel(model, tokenizer, optimizer, lr_scheduler)
save_dir = os.path.join(args.save_dir, experiment_name)
callbacks = [L.pytorch.callbacks.ModelCheckpoint(dirpath=save_dir, 
                                                 filename=args.run_name + '-{epoch}', 
                                                 monitor='loss_dev', mode='min', save_top_k=3),
             L.pytorch.callbacks.EarlyStopping(monitor='loss_dev', mode='min', patience=3)]
logger = L.pytorch.loggers.MLFlowLogger(save_dir='logs', experiment_name=experiment_name, run_name=args.run_name)
trainer = L.Trainer(max_epochs=args.epochs,
                    logger=logger,
                    callbacks=callbacks)
trainer.fit(model, train_loader, dev_loader)
