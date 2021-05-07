#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH
#!
DATASET=game_mt
MODEL=plain
PENCODING=absolute

CUDA_VISIBLE_DEVICES=0 python main.py \
 --dir-path ../data/$DATASET/2ko/encoded_mapped \
 --loss-type plain \
 --positional-encoding $PENCODING \
 --task mnmt \
 --target-lang ko \
 --model-type $MODEL \
 --dataset-name $DATASET \
 --model-size base;
