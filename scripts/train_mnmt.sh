#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH
#!
DATASET=un_pc
MODEL=plain
PENCODING=absolute

CUDA_VISIBLE_DEVICES=0 python main.py \
 --dir-path ../data/$DATASET/multi/encoded_mapped \
 --loss-type plain \
 --positional-encoding $PENCODING \
 --task mnmt \
 --model-type $MODEL \
 --dataset-name $DATASET \
 --model-size base;
