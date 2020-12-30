#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CKPT=epoch_0_ckpt_1
LR=0.00011

CUDA_VISIBLE_DEVICES=0 python text_sample.py \
  --task multitask \
  --is-sampling \
  --dir-path ../data/multitask \
  --tokenizer-prefix WP_30000 \
  --saved-model-folder data/saved_model/multitask/base_$LR \
  --saved-model-ckpt $CKPT \
  --sampling-mode beam \
  --target-text-path ../data/multitask/temp.txt \
  --width 4 \
  --lengths-penalty 1 \
  --model-size base;
