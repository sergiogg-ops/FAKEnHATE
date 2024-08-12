from transformers import AutoTokenizer, RobertaModel
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import sys
import re

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if len(sys.argv) != 2:
    print('Uso: python add.py num')
    exit()
N = int(sys.argv[1])

fake = pd.read_json('../data/pubreleasednewsfiles/full.json')
#true = pd.read_json('data/LOCO/subset_mainstream.json')
true = pd.read_json('../data/LOCO/LOCO_sel_trans.json')
base = pd.read_json('../data/base/train.json')

if len(fake) < N or len(true) < N:
    # max 6942
    print(f'Not enough data:\n\tFake:{len(fake)}\n\tTrue:{len(true)}')
    exit()

# PONDERATION
tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
model = RobertaModel.from_pretrained("PlanTL-GOB-ES/roberta-base-bne").to(device)
def get_embeddings(texts, model, tokenizer, batch_size=8):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Embedding'):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(inputs['input_ids'].to(device), inputs['attention_mask'].to(device)).last_hidden_state
            batch_embs = torch.mean(outputs, dim=1).cpu()
        embs.append(batch_embs)
        torch.cuda.empty_cache()
    return torch.cat(embs).numpy()
fake_embs = get_embeddings(fake['text'].tolist(), model, tokenizer)
true_embs = get_embeddings(true['text'].tolist(), model, tokenizer)
#base_embs = get_embeddings(base['text'].tolist(), model, tokenizer)
fake_center = np.mean(fake_embs)
#fake_center = np.mean(base_embs[base['category'] == 'Fake'])
true_center = np.mean(true_embs)
#true_center = np.mean(base_embs[base['category'] == 'True'])
fake_weights = np.linalg.norm(fake_embs - fake_center,axis=1)
true_weights = np.linalg.norm(true_embs - true_center,axis=1)

# SELECT N SAMPLES
def mask_entities(df):
    df['text'] = df['text'].apply(lambda x: re.sub(r'\b(?:\+?1[-.\s]?)?(?:\(?[2-9]\d{2}\)?[-.\s]?){1,2}\d{4}\b','*PHONE*',x))
    df['headline'] = df['headline'].apply(lambda x: re.sub(r'\b(?:\+?1[-.\s]?)?(?:\(?[2-9]\d{2}\)?[-.\s]?){1,2}\d{4}\b','*PHONE*',x))
    df['text'] = df['text'].apply(lambda x: re.sub(r'\b(?:https?://)?(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b','*URL*',x))
    df['headline'] = df['headline'].apply(lambda x: re.sub(r'\b(?:https?://)?(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b','*URL*',x))
    df['text'] = df['text'].apply(lambda x: re.sub(r'\d+','*NUMBER*',x))
    df['headline'] = df['headline'].apply(lambda x: re.sub(r'\d+','*NUMBER*',x))
    return df
fake = mask_entities(fake.sample(N,weights=fake_weights))
true = mask_entities(true.sample(N,weights=true_weights))

# ADJUST THE FIELD NAMES
fake.rename(columns={'label':'category'}, inplace=True)
fake['category'] = 'Fake'
#true = true[true['subcorpus'] == 'mainstream']
true.rename(columns={'subcorpus':'category'}, inplace=True)
true['category'] = 'True'

# LENGTHS BIAS CHECK
lengths = [len(sample['text'].split()) for _,sample in fake.iterrows()]
print(np.mean(lengths),np.std(lengths))
print(np.mean(np.array(true['txt_nwords'])),np.std(np.array(true['txt_nwords'])))

# SAVE DATA
base = pd.concat((base, fake, true))
base.to_json('../data/ext_data/train.json',orient='records')