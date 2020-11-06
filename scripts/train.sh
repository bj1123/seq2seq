#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python lm_main.py \
    --filpath temp.npy \
    --model-size base;
