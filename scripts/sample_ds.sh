#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

DATASET=wikilarge
CKPT=epoch_5
LR=0.00011

CUDA_VISIBLE_DEVICES=0 python sample.py \
  --src-path ../data/data-simplification/$DATASET/splited/src_encoded_mapped \
  --tgt-path ../data/data-simplification/$DATASET/splited/tgt_encoded_mapped \
  --saved-model-folder data/saved_model/$DATASET/base_$LR \
  --saved-model-ckpt $CKPT \
  --sample-save-path data/sampled/$DATASET/base_$LR/$CKPT \
  --sampling-mode beam \
  --dataset-name $DATASET \
  --width 8 \
  --lengths-penalty 1 \
  --model-size base;
