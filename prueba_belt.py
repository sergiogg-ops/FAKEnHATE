import utils
from transformers import RobertaModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
model = utils.FakeBELT(RobertaModel.from_pretrained("PlanTL-GOB-ES/roberta-base-bne"), tokenizer=tokenizer, add_noise=False)

train_set = utils.FakeSet('data/base/train.json', sep=tokenizer.sep_token, verbose=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1)

output = model(next(iter(train_loader))['text'])
print(output)