#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

DATASET=wmt

CUDA_VISIBLE_DEVICES=0 python main.py \
 --src-path ../data/$DATASET/en_encoded \
 --tgt-path ../data/$DATASET/de_encoded \
 --loss-type plain \
 --positional-encoding absolute \
 --task seq2seq \
 --model-type sentence-aware \
 --dataset-name $DATASET \
 --model-size base;
