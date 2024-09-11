!/bin/bash

#python train.py data/original/ models/ -size large -lr 2e-6 -lr_sch 0.7 -exp original_large -run base -b 2 -acc 4 -val_interval 676
#python train.py data/original/ models/ -size large -lr 2e-6 -lr_sch 0.7 -exp original_large -run PER-ORG -b 2 -acc 4 -ner PER ORG -val_interval 676
python train.py data/backtranslation/ models/ -lr 2e-5 -lr_sch 0.7 -exp original -run pipe5 -val_interval 676
python train.py data/original/ models/ -lr 2e-5 -lr_sch 0.7 -exp original -run noisy_emb -no uniform -alpha 10 -val_interval 676