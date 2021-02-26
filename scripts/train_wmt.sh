#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

DATASET=wmt
MODEL=sentence-aware
PENCODING=absolute

CUDA_VISIBLE_DEVICES=0 python main.py \
 --src-path ../data/$DATASET/en_encoded \
 --tgt-path ../data/$DATASET/de_encoded \
 --loss-type plain \
 --positional-encoding $PENCODING \
 --task seq2seq \
 --model-type $MODEL \
 --dataset-name $DATASET \
 --model-size base;
