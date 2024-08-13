!/bin/bash

python train.py data/base/ models/ -lr 5e-6 -lr_sch 0.7 -exp noisy_emb -no uniform -alpha 0.1 -run uniform_1e-1
python train.py data/base/ models/ -lr 5e-6 -lr_sch 0.7 -exp noisy_emb -no uniform -alpha 1 -run uniform_1
python train.py data/base/ models/ -lr 5e-6 -lr_sch 0.7 -exp noisy_emb -no uniform -alpha 5 -run uniform_5
python train.py data/base/ models/ -lr 5e-6 -lr_sch 0.7 -exp noisy_emb -no uniform -alpha 10 -run uniform_10
python train.py data/base/ models/ -lr 5e-6 -lr_sch 0.7 -exp noisy_emb -no uniform -alpha 15 -run uniform_15
python train.py data/base/ models/ -lr 5e-6 -lr_sch 0.7 -exp noisy_emb -no uniform -alpha 20 -run uniform_20
python train.py data/base/ models/ -lr 5e-6 -lr_sch 0.7 -exp noisy_emb -no uniform -alpha 25 -run uniform_25
python train.py data/base/ models/ -lr 5e-6 -lr_sch 0.7 -exp noisy_emb -no uniform -alpha 50 -run uniform_50

python train.py data/base/ models/ -lr 5e-6 -lr_sch 0.7 -exp noisy_emb -no normal -alpha 1 -run normal_1
python train.py data/base/ models/ -lr 5e-6 -lr_sch 0.7 -exp noisy_emb -no normal -alpha 0.1 -run normal_1e-1
python train.py data/base/ models/ -lr 5e-6 -lr_sch 0.7 -exp noisy_emb -no normal -alpha 0.01 -run normal_1e-2
python train.py data/base/ models/ -lr 5e-6 -lr_sch 0.7 -exp noisy_emb -no normal -alpha 0.05 -run normal_5e-2
python train.py data/base/ models/ -lr 5e-6 -lr_sch 0.7 -exp noisy_emb -no normal -alpha 5 -run normal_5
python train.py data/base/ models/ -lr 5e-6 -lr_sch 0.7 -exp noisy_emb -no normal -alpha 10 -run normal_10