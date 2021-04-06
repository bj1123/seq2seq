#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH
#!
DATASET=un_pc
MODEL=language-specific
PENCODING=absolute

CUDA_VISIBLE_DEVICES=0 python main.py \
 --dir-path ../data/$DATASET/multi/encoded_mapped \
 --loss-type plain \
 --positional-encoding $PENCODING \
 --task mnmt \
 --target-lang en \
 --model-type $MODEL \
 --dataset-name $DATASET \
 --model-size base;
