#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python main.py \
 --src-path ../data/wmt/en_encoded \
 --tgt-path ../data/wmt/de_encoded \
 --loss-type label-smoothing \
 --model-size base;
