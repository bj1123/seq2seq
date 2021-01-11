#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

DATASET=wikilarge

CUDA_VISIBLE_DEVICES=0 python main.py \
--src-path ../data/data-simplification/$DATASET/splited/src_encoded_mapped \
--tgt-path ../data/data-simplification/$DATASET/splited/tgt_encoded_mapped \
--prob-path ../data/data-simplification/$DATASET/splited/wp_tgt_10000-probs.json \
--dataset-name $DATASET \
--loss-type plain \
--model-type sentence-aware \
--task seq2seq \
--model-size base;
