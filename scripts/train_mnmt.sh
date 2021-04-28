#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH
#!
DATASET=game_mt
MODEL=plain
PENCODING=absolute

CUDA_VISIBLE_DEVICES=0 python main.py \
 --dir-path ../data/$DATASET/frac/encoded_mapped_ml \
 --loss-type plain \
 --positional-encoding $PENCODING \
 --task mnmt \
 --target-lang en \
 --model-type $MODEL \
 --dataset-name $DATASET \
 --model-size base;
