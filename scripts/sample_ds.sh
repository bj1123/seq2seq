#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

DATASET=wikilarge
CKPT=epoch_7
LR=0.0001

CUDA_VISIBLE_DEVICES=0 python sample.py \
  --src-path ../data/data-simplification/$DATASET/raw_encoded_mapped/src \
  --tgt-path ../data/data-simplification/$DATASET/raw_encoded_mapped/tgt \
  --saved-model-folder data/saved_model/$DATASET/base_$LR \
  --saved-model-ckpt $CKPT \
  --sample-save-path data/sampled/$DATASET/base_$LR/$CKPT \
  --sampling-mode beam \
  --dataset-name $DATASET \
  --width 8 \
  --lengths-penalty 0.2 \
  --model-size base;
