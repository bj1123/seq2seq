#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

DATASET=game_mt/raw/ja2ko
MODEL=plain
PENCODING=absolute

CUDA_VISIBLE_DEVICES=0 python main.py \
 --src-path ../data/$DATASET/splited/ja/encoded_mapped \
 --tgt-path ../data/$DATASET/splited/ko/encoded_mapped \
 --loss-type plain \
 --positional-encoding $PENCODING \
 --task seq2seq \
 --model-type $MODEL \
 --dataset-name $DATASET \
 --model-size base;
