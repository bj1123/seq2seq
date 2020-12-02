#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

DATASET=wikilarge

CUDA_VISIBLE_DEVICES=0 python main.py \
 --src-path ../data/data-simplification/$DATASET/splited_encoded_mapped/src \
 --tgt-path ../data/data-simplification/$DATASET/splited_encoded_mapped/tgt \
 --dataset-name $DATASET \
 --loss-type plain \
 --model-size base;
