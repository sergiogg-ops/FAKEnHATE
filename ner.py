import subprocess

opts = ['PER','MISC','ORG','LOC']

for i in range(len(opts)):
    for j in range(i+1,len(opts)):
        subprocess.run(f'python train.py data/ models/ -ner {opts[i]} {opts[j]} -run {opts[i]}-{opts[j]} -v', shell=True)