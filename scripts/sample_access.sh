#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CKPT=epoch_22
MODEL=sentence-aware
LR=0.000109

CUDA_VISIBLE_DEVICES=0 python sample.py \
  --is-sampling \
  --task access \
  --dataset _2912c535c2343258d2e6375bca3e3a3d \
  --saved-model-folder data/saved_model/access/$MODEL/base_$LR \
  --saved-model-ckpt $CKPT \
  --sample-save-path data/sampled/access/$MODEL/base_$LR/$CKPT \
  --sampling-mode beam \
  --model-type $MODEL \
  --width 8 \
  --lengths-penalty 1 \
  --model-size base;
