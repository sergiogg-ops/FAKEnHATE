from transformers import RobertaModel, AutoTokenizer
from argparse import ArgumentParser
from sklearn.metrics import classification_report, roc_curve
import lightning as L
import pandas as pd
import os
import utils
import torch

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
utils.seed_everything(42)

parser = ArgumentParser(description='Evaluate a BERT like model for the task of fake news detection.')
parser.add_argument('file', help='File to evaluate')
parser.add_argument('model', help='Model checkpoint')
parser.add_argument('-b','--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('-ner','--mask_ner', nargs='*', help='Mask named entities, if no arguments are given all entities are masked')
parser.add_argument('-v','--verbose', default=False, action='store_true', help='Verbose mode: detailed statistics and a hint of an optimum classification threshold')
parser.add_argument('-out','--output', help='If provided the file in which the predictions will be saved')
parser.add_argument('-size','--model_size', default='base', choices=['base','large'], help='Model size')
parser.add_argument('-full','--full_length', default=False, action='store_true', help='Use full length of the text')
parser.add_argument('-pool','--pool_strategy', default='transf',choices=['max','avg','sum','attn','rnn','transf'], help='Aggregation strategy for the CLS tokens of each chunk.')
parser.add_argument('-thr','--threshold', type=float, default=0.5, help='Threshold for the fake news detection')
args = parser.parse_args()

#tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
#model = RobertaModel.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
tokenizer = AutoTokenizer.from_pretrained(f"PlanTL-GOB-ES/roberta-{args.model_size}-bne")
model = RobertaModel.from_pretrained(f"PlanTL-GOB-ES/roberta-{args.model_size}-bne")

if args.full_length:
    model = utils.FakeBELT(model, tokenizer=tokenizer, pool=args.pool_strategy)
    #model = utils.CustomBELT(model, tokenizer=tokenizer)
else:
    model = utils.FakeModel(model, tokenizer=tokenizer)

model = utils.LightningModel.load_from_checkpoint(args.model, model=model, tokenizer=tokenizer)
masker = utils.NamedEntityMasker(args.mask_ner) if args.mask_ner else None

data = pd.read_json(args.file)
test_set = utils.FakeSet(args.file, sep=tokenizer.sep_token, masker=masker, verbose=True)
num_workers = os.cpu_count() - 1
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

predictions = L.Trainer(logger=False).predict(model, dataloaders=[test_loader], ckpt_path=args.model)
predictions = torch.nn.functional.softmax(torch.concatenate(predictions), dim=1)

fake_scores = predictions[:,0].tolist()
conf = torch.sqrt(torch.mean(torch.square(predictions[:,0] - 0.5)))
predictions = torch.where(predictions[:,0] >= args.threshold, 0, 1)
if args.output:
    data['prob_fake'] = fake_scores
    data['predictions'] = [utils.ID2LABEL[l.item()] for l in predictions]
    data.to_json(args.output, orient='records')

print(classification_report([item['label'] for item in test_set], predictions, 
                            target_names=utils.ID2LABEL.values(),
                            digits=4))

if args.verbose:
    stats = {}
    predictions = predictions
    for i in range(len(test_set)):
        if data['topic'][i] not in stats:
            stats[data['topic'][i]] = {'pred':[predictions[i].item()], 'true':[test_set[i]['label'].item()]}
        else:
            stats[data['topic'][i]]['pred'].append(predictions[i].item())
            stats[data['topic'][i]]['true'].append(test_set[i]['label'].item())

    for topic, values in stats.items():
        print('#####################################################')
        print(f'#Topic: {topic}')
        print('#####################################################')
        print(classification_report(values['true'], values['pred'],
                                    labels=torch.arange(len(utils.ID2LABEL)).tolist(),
                                    target_names=utils.ID2LABEL.values(),
                                    digits=4,
                                    zero_division=0))
    
    fpr, tpr, thrs = roc_curve([item['label'] for item in test_set], fake_scores, pos_label=0)
    best = torch.argmin(torch.sqrt(torch.tensor(fpr[1:-1]**2) + (torch.tensor((1-tpr[1:-1])**2)))) + 1
    print(f'Optimum threshold: {thrs[best]}')
print(f'Mean confidence: {conf.item()}')