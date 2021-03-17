#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

DATASET=un_pc
MODEL=plain
TYPE=multi
PAIR=ar-en
PENCODING=absolute
LR=0.00021
CKPT=epoch_1

CUDA_VISIBLE_DEVICES=0 python sample.py \
  --is-sampling \
  --task seq2seq \
  --src-path ../data/$DATASET/$TYPE/encoded_mapped/$PAIR/ar \
  --tgt-path ../data/$DATASET/$TYPE/encoded_mapped/$PAIR/en \
  --saved-model-folder data/saved_model/$DATASET/$TYPE/$MODEL/base_$LR''_$PENCODING \
  --saved-model-ckpt $CKPT \
  --sample-save-path data/sampled/$DATASET/$TYPE/$PAIR/$MODEL/base_$LR''_$PENCODING/$CKPT \
  --sampling-mode beam \
  --positional-encoding $PENCODING \
  --dataset-name $DATASET \
  --model-type $MODEL \
  --width 4 \
  --lengths-penalty 0.6 \
  --model-size base;
