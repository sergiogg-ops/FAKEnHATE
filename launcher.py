import subprocess

for s in [0.5, 0.7, 0.8, 0.9]:
    subprocess.run(f'python train.py data/base models/ -lr 2e-5 -lr_sch 0.7 -full -pool max -v -exp full_length -run max_step{s} -b 2 -s {s}'.split())