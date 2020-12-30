#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

DATASET=wikilarge
CKPT=epoch_9
LR=0.00011
MODEL=complexity

CUDA_VISIBLE_DEVICES=0 python sample.py \
  --is-sampling \
  --src-path ../data/data-simplification/$DATASET/splited/src_encoded_mapped \
  --tgt-path ../data/data-simplification/$DATASET/splited/tgt_encoded_mapped \
  --saved-model-folder data/saved_model/$DATASET/$MODEL/base_$LR \
  --saved-model-ckpt $CKPT \
  --sample-save-path data/sampled/$DATASET/$MODEL/base_$LR/$CKPT \
  --sampling-mode beam \
  --dataset-name $DATASET \
  --width 8 \
  --task seq2seq \
  --lengths-penalty 1 \
  --prob-path ../data/data-simplification/$DATASET/splited/wp_tgt_10000-probs.json \
  --complexity-aware \
  --model-size base;
