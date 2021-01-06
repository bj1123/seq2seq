#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CKPT=epoch_50
LR=0.000111

CUDA_VISIBLE_DEVICES=0 python sample.py \
  --is-sampling \
  --task access \
  --dataset _2912c535c2343258d2e6375bca3e3a3d \
  --saved-model-folder data/saved_model/access/base_$LR \
  --saved-model-ckpt $CKPT \
  --sample-save-path data/sampled/access/base_$LR/$CKPT \
  --sampling-mode beam \
  --width 8 \
  --lengths-penalty 1 \
  --model-size base;
