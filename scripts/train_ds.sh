#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

DATASET=wikilarge

CUDA_VISIBLE_DEVICES=0 python main.py \
--src-path ../data/data-simplification/$DATASET/splited/src_encoded_mapped \
--tgt-path ../data/data-simplification/$DATASET/splited/tgt_encoded_mapped \
--dataset-name $DATASET \
--loss-type plain \
--model-size base;
