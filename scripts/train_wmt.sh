#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

DATASET=wmt

CUDA_VISIBLE_DEVICES=0 python main.py \
 --src-path ../data/$DATASET/en_encoded_splited \
 --tgt-path ../data/$DATASET/de_encoded_splited \
 --loss-type plain \
 --positional-encoding relative \
 --task seq2seq \
 --model-type plain \
 --dataset-name $DATASET \
 --model-size base;
