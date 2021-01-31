#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CKPT=epoch_8
LR=0.000111

CUDA_VISIBLE_DEVICES=0 python text_sample.py \
  --task multitask \
  --is-sampling \
  --dir-path ../data/multitask_access \
  --tokenizer-prefix wp_30000 \
  --saved-model-folder data/saved_model/multitask/base_$LR \
  --saved-model-ckpt $CKPT \
  --sampling-mode beam \
  --target-text-path ../data/multitask_access/temp.txt \
  --width 8 \
  --lengths-penalty 1 \
  --model-size base;
