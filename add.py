import pandas as pd
import numpy as np
import sys

np.random.seed(42)

if len(sys.argv) != 2:
    print('Uso: python add.py num')
    exit()
N = int(sys.argv[1])

fake = pd.read_json('data/pubreleasednewsfiles/full.json')
true = pd.read_json('data/LOCO/subset_mainstream.json')
base = pd.read_json('data/base/train.json')

if len(fake) < N or len(true) < N:
    # max 6942
    print(f'Not enough data:\n\tFake:{len(fake)}\n\tTrue:{len(true)}')
    exit()

# SELECT N SAMPLES
fake_idx = np.random.randint(len(fake), size=N)
true_idx = np.random.randint(len(true), size=N)
fake = pd.DataFrame([fake.loc[fake_idx[i]] for i in range(N)])
true = pd.DataFrame([true.loc[true_idx[i]] for i in range(N)])
#mean_length = np.mean([len(sample['text'].split()) for _,sample in fake.iterrows()])
#weights = 1 / np.abs(np.array(true['txt_nwords'],dtype=int) - mean_length + 0.1)
#true = true.sample(n=N, weights=weights, random_state=seed)
#true.to_json('LOCO/subset_true.json',orient='records')
#exit()

# ADJUST THE FIELD NAMES
fake.rename(columns={'label':'category'}, inplace=True)
fake['category'] = 'Fake'
#true = true[true['subcorpus'] == 'mainstream']
#true.rename(columns={'subcorpus':'category','txt':'text'}, inplace=True)
#true['category'] = 'True'

# LENGTHS BIAS CHECK
lengths = [len(sample['text'].split()) for _,sample in fake.iterrows()]
print(np.mean(lengths),np.std(lengths))
print(np.mean(np.array(true['txt_nwords'])),np.std(np.array(true['txt_nwords'])))

# SAVE DATA
base = pd.concat((base, fake, true))
base.to_json('data/ext_data/train.json',orient='records')