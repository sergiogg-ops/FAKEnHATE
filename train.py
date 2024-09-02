from transformers import RobertaModel, AutoTokenizer
from argparse import ArgumentParser
import lightning as L
import os
import utils
import torch

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
utils.seed_everything(42)

parser = ArgumentParser()
parser.add_argument('data_dir', help='Directory of the traning data')
parser.add_argument('save_dir', help='Directory to save the model')
parser.add_argument('-e','--epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('-b','--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('-lr','--learning_rate', type=float, default=2e-5, help='Learning rate')
parser.add_argument('-exp','--experiment_name', default='baseline', help='Name of the series of runs')
parser.add_argument('-run','--run_name', type=str, default='run', help='Name of the run')
parser.add_argument('-ner','--mask_ner', nargs='*', help='Mask named entities, if no arguments are given all entities are masked')
parser.add_argument('-lr_sch','--lr_scheduler', default=0.7, type=float, help='Start factor for the linear scheduler')
parser.add_argument('-no','--noise', default=False, choices=['uniform','normal'], help='Use noisy embeddings')
parser.add_argument('-alpha','--alpha',type=float, default=0, help='Alpha paratemetr to scale the noise in the embeddings')
parser.add_argument('-full','--full_length', default=False, action='store_true', help='Use full length of the text')
parser.add_argument('-pool','--pool_strategy', default='max',choices=['max','avg','sum','attn','rnn','transf'], help='Aggregation strategy for the CLS tokens of each chunk.')
parser.add_argument('-s','--stride',type=float, default=0.75,help='Stride for the full length proccessing approach')
parser.add_argument('-unfreeze','--unfreeze_epoch',type=int,default=0,help='When using the full length apprach, the epoch in which the params of the BERT type model will unfreeze. If not especified the params will be unfrozen from the start.')
parser.add_argument('-t','--test', default=False, action='store_true', help='Test the model at the end of training.')
parser.add_argument('-v','--verbose', default=False, action='store_true', help='Verbose mode')
args = parser.parse_args()

if args.mask_ner == [] and args.verbose:
        print('Todas las entidades serán enmascaradas')

tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
model = RobertaModel.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
if args.full_length:
        model = utils.FakeBELT(model, tokenizer=tokenizer, add_noise=args.noise, max_length=2500, pool=args.pool_strategy, step=args.stride)
else:
        model = utils.FakeModel(model, tokenizer=tokenizer, add_noise=args.noise)

masker = utils.NamedEntityMasker(args.mask_ner) if args.mask_ner else None

train_set = utils.FakeSet(os.path.join(args.data_dir,'train.json'), sep=tokenizer.sep_token, masker=masker, verbose=args.verbose)
dev_set = utils.FakeSet(os.path.join(args.data_dir,'dev.json'), sep=tokenizer.sep_token, masker=masker, verbose=args.verbose)
if args.test:
        test_set = utils.FakeSet(os.path.join(args.data_dir,'test.json'), sep=tokenizer.sep_token, masker=masker, verbose=args.verbose)

num_workers = os.cpu_count() - 1
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
if args.test:
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

#optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9,0.999), eps=1e-8)
#lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = args.lr_scheduler, end_factor = 1, total_iters = args.epochs)

model = utils.LightningModel(model, tokenizer, 
                             lr=args.learning_rate, 
                             sch_factor=args.lr_scheduler, 
                             sch_iters=args.epochs, 
                             alpha=args.alpha,
                             unfreeze=args.unfreeze_epoch)
save_dir = os.path.join(args.save_dir, args.experiment_name)
callbacks = [L.pytorch.callbacks.ModelCheckpoint(dirpath=save_dir, 
                                                 filename=args.run_name + '-{step}', 
                                                 monitor='f1_dev', mode='max', save_top_k=2),
             L.pytorch.callbacks.EarlyStopping(monitor='loss_dev', mode='min', patience=3)]
logger = L.pytorch.loggers.MLFlowLogger(save_dir='logs', experiment_name=args.experiment_name, run_name=args.run_name)
trainer = L.Trainer(max_steps=int(987/args.batch_size+1)*25,
                    logger=logger,
                    callbacks=callbacks,
                    check_val_every_n_epoch=None,
                    val_check_interval=int(987/args.batch_size+1),
                    accumulate_grad_batches=1)

trainer.fit(model, train_loader, dev_loader)
if args.test:
        trainer.test(model, test_loader, ckpt_path='best')