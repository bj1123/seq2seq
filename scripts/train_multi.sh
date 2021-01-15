#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python main.py \
--task multitask \
--dir-path ../data/multitask_access \
--tokenizer-prefix wp_30000 \
--loss-type plain \
--model-size base;
