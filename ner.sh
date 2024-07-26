
python train.py data/ models/ -b 16 -lr 5e-6 -lr_sch 0.6 -ner MISC ORG --run MISC-ORG-conf6
python train.py data/ models/ -b 16 -lr 5e-6 -lr_sch 0.6 -ner ORG LOC --run ORG-LOC-conf6