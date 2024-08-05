from transformers import AutoTokenizer, RobertaModel
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import sys

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if len(sys.argv) != 2:
    print('Uso: python add.py num')
    exit()
N = int(sys.argv[1])

true = pd.read_json('data/LOCO/LOCO.json')
true = true[true['subcorpus'] == 'mainstream']

# PONDERATION
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
model = RobertaModel.from_pretrained("FacebookAI/roberta-base").to(device)
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
def get_subset(data, n, model, tokenizer, seed=42, batch_size=8):
    emb = get_embeddings(data['txt'].tolist(), model, tokenizer)
    data['embedding'] = [np.array(e) for e in emb]
    res = data.sample(1, random_state=seed)
    for _ in tqdm(range(0,n,batch_size),desc='Creando subset'):
        centroid = KMeans(n_clusters=1, random_state=seed).fit(np.stack(res['embedding'])).cluster_centers_
        weights = np.linalg.norm(np.stack(data['embedding']) - centroid,axis=1)
        res = pd.concat((res, data.sample(batch_size,weights=weights)))
    return res
true = get_subset(true, N, model, tokenizer, 50)

# SAVE DATA
true.to_json('data/LOCO/LOCO_selection.json',orient='records')