#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python main.py \
--task multitask \
--dir-path ../data/multitask \
--tokenizer-prefix WP_30000 \
--loss-type plain \
--model-size base;
