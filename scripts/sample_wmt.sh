#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

DATASET=wmt
MODEL=plain
PENCODING=absolute
LR=0.00021
CKPT=epoch_9

CUDA_VISIBLE_DEVICES=0 python sample.py \
  --is-sampling \
  --task seq2seq \
  --src-path ../data/wmt/en_encoded \
  --tgt-path ../data/wmt/de_encoded \
  --saved-model-folder data/saved_model/$DATASET/$MODEL/base_$LR''_$PENCODING \
  --saved-model-ckpt $CKPT \
  --sampling-mode beam \
  --positional-encoding $PENCODING \
  --dataset-name $DATASET \
  --model-type $MODEL \
  --width 4 \
  --lengths-penalty 0.6 \
  --model-size base;
