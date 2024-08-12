import numpy as np
import yaml
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('dir', help='Directory of the experiment')
parser.add_argument('experiment', help='Directory of the experiment')
parser.add_argument('metric', help='Metric to compare')
parser.add_argument('-n','--top', type=int, default=5, help='Number of top runs to show')
parser.add_argument('-m','--mode', default='max', choices=['max','min'], help='Mode of the metric')
args = parser.parse_args()

exp2id = {}
for exp in os.listdir(args.dir):
    try:
        with open(os.path.join(args.dir, exp, 'meta.yaml')) as f:
            exp2id[yaml.safe_load(f)['name']] = exp
    except:
        pass

try:
    runs = os.listdir(os.path.join(args.dir,exp2id[args.experiment]))
except:
    print('Experiment not found\nAvailable experiments:')
    print(list(exp2id.keys()))
    exit(1)
results = []
names = []

for run in runs:
    try:
        with open(os.path.join(args.dir,exp2id[args.experiment], run, 'metrics', args.metric)) as f:
            data = [float(l.split()[1]) for l in f]
        results.extend(data)
        with open(os.path.join(args.dir,exp2id[args.experiment], run, 'meta.yaml')) as f:
            n = yaml.safe_load(f)['run_name']
        names.extend([n]*len(data))
    except:
        pass

idxs = np.argsort(results)
if args.mode == 'max':
    idxs = idxs[::-1]
for i in idxs[:args.top]:
    print(f'{names[i]}: {results[i]}')