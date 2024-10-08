# Adapt the FakeNewsCorpusSpanish dataset to a uniform format.

import pandas as pd
import os

train = pd.read_excel('data/FakeNewsCorpusSpanish/train.xlsx')
mapping = {}
for col in train.columns:
    mapping[col] = col.lower()
train.rename(columns=mapping, inplace=True)
dev = pd.read_excel('data/FakeNewsCorpusSpanish/development.xlsx').rename(columns=mapping)

for col in train.columns:
    mapping[col.upper()] = col
test = pd.read_excel('data/FakeNewsCorpusSpanish/test.xlsx').rename(columns=mapping)
test.rename(columns={'TOPICS':'topic'}, inplace=True)
test['category'] = test['category'].apply(lambda x: 'True' if x else 'Fake')

train['id'] = train['id'].apply(lambda x: f'fncs_train_{x}')
dev['id'] = dev['id'].apply(lambda x: f'fncs_dev_{x}')
test['id'] = test['id'].apply(lambda x: f'fncs_test_{x}')

if not os.path.exists('data/original'):
    os.makedirs('data/original')
train.to_json('data/original/train.json', orient='records')
dev.to_json('data/original/dev.json', orient='records')
test.to_json('data/original/test.json', orient='records')