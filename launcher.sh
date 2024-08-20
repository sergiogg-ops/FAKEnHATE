#!/bin/bash

python train.py data/base models/ -lr 2e-5 -lr_sch 0.7 -full -pool max -v -exp full_length -run max_pool -b 2
python train.py data/base models/ -lr 2e-5 -lr_sch 0.7 -full -pool avg -v -exp full_length -run avg_pool -b 2
python train.py data/base models/ -lr 2e-5 -lr_sch 0.7 -full -pool sum -v -exp full_length -run sum_pool -b 2