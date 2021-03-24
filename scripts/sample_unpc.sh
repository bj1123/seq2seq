#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

DATASET=un_pc
MODEL=plain
DATATYPE=mono
TRAINING_TYPE=semi/en
PAIR=ar-en
PENCODING=absolute
LR=0.00021
CKPT=epoch_9

CUDA_VISIBLE_DEVICES=0 python sample.py \
  --is-sampling \
  --task seq2seq \
  --src-path ../data/$DATASET/$DATATYPE/$PAIR/encoded_mapped/ar \
  --tgt-path ../data/$DATASET/$DATATYPE/$PAIR/encoded_mapped/en \
  --saved-model-folder data/saved_model/$DATASET/$MODEL/base_$LR''_$PENCODING \
  --saved-model-ckpt $CKPT \
  --sampling-mode beam \
  --positional-encoding $PENCODING \
  --dataset-name $DATASET \
  --model-type $MODEL \
  --width 4 \
  --lengths-penalty 0.6 \
  --model-size base;
