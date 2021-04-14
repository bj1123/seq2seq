#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

DATASET=game_mt
MODEL=plain
DATATYPE=frac
TRAINING_TYPE=semi/en
PAIR=zh2en
PENCODING=absolute
LR=0.00021
CKPT=epoch_2

CUDA_VISIBLE_DEVICES=0 python sample.py \
  --is-sampling \
  --task seq2seq \
  --src-path ../data/$DATASET/$DATATYPE/encoded_mapped/$PAIR/zh \
  --tgt-path ../data/$DATASET/$DATATYPE/encoded_mapped/$PAIR/en \
  --saved-model-folder data/saved_model/$DATASET/$TRAINING_TYPE/$MODEL/base_$LR''_$PENCODING \
  --saved-model-ckpt $CKPT \
  --sampling-mode beam \
  --positional-encoding $PENCODING \
  --dataset-name $DATASET \
  --model-type $MODEL \
  --width 4 \
  --lengths-penalty 0.6 \
  --model-size base;
