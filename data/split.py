import pandas as pd
from sklearn.model_selection import train_test_split

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

data = pd.concat([train, test, dev])
data['stratification'] = str(data['category']) + '/' + str(data['topic'])

train, test = train_test_split(data, test_size=0.2, stratify=data['stratification'], random_state=42)
train, dev = train_test_split(train, test_size=0.2, stratify=train['stratification'], random_state=42)

train.drop(columns='stratification', inplace=True)
dev.drop(columns='stratification', inplace=True)
test.drop(columns='stratification', inplace=True)

train.to_json('data/train.json', orient='records')
dev.to_json('data/dev.json', orient='records')
test.to_json('data/test.json', orient='records')