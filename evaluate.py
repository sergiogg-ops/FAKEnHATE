from transformers import RobertaModel, AutoTokenizer
from argparse import ArgumentParser
from sklearn.metrics import classification_report
from tqdm import tqdm
import lightning as L
import pandas as pd
import os
import utils
import torch

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
utils.seed_everything(42)

parser = ArgumentParser()
parser.add_argument('file', help='File to evaluate')
parser.add_argument('model', help='Model checkpoint')
parser.add_argument('-b','--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('-ner','--mask_ner', nargs='*', help='Mask named entities, if no arguments are given all entities are masked')
parser.add_argument('-v','--verbose', default=False, action='store_true', help='Verbose mode')
parser.add_argument('-out','--output', help='If provided the file in which the predictions will be saved')
parser.add_argument('-full','--full_length', default=False, action='store_true', help='Use full length of the text')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
model = RobertaModel.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
if args.full_length:
    model = utils.FakeBELT(model, tokenizer=tokenizer)
else:
    model = utils.FakeModel(model, tokenizer=tokenizer)

model = utils.LightningModel.load_from_checkpoint(args.model, model=model, tokenizer=tokenizer)
masker = utils.NamedEntityMasker(args.mask_ner) if args.mask_ner else None

test_set = utils.FakeSet(args.file, sep=tokenizer.sep_token, masker=masker, verbose=args.verbose)
num_workers = os.cpu_count() - 1
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

predictions = L.Trainer(logger=False).predict(model, dataloaders=[test_loader], ckpt_path=args.model)
# predictions = []
# for batch in tqdm(test_loader, desc='Prediciendo', total=len(test_loader)):
#     predictions.append(model.forward(batch['text']))
predictions = torch.nn.functional.softmax(torch.concatenate(predictions))
if args.output:
    data = pd.read_json(args.file)
    data['predictions'] = [utils.ID2LABEL[l.item()] for l in predictions]
    data.to_json(args.output, orient='records')

print(classification_report([item['label'] for item in test_set], predictions.argmax(dim=1), 
                            target_names=utils.ID2LABEL.values(),
                            digits=4))
conf = torch.mean(torch.abs(predictions[0:] - 0.5))
print(f'Mean confidence: {conf.item()}')

stats = {}
for i in range(len(test_set)):
    if test_set[i]['topic'] not in stats:
        stats[test_set[i]['topic']] = {'pred':predictions[i], 'true':test_set[i]['label']}
    else:
        stats[test_set[i]['topic']]['pred'] += predictions[i]
        stats[test_set[i]['topic']]['true'] += test_set[i]['label']

for topic, values in stats.items():
    print(f'Topic: {topic}')
    print(classification_report([values['true']], values['pred'].argmax(dim=0), 
                                target_names=utils.ID2LABEL.values(),
                                digits=4))