#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

Mode=unlikelihood-token-seq
Data=wiki103
CUDA_VISIBLE_DEVICES=0 python lm_ppl_eval.py \
    --saved-path data/$Data/_$Mode''_layer_12_lr_0.0001_cutoffs_18_nbar_14 \
    --dataset $Data \
    --loss-type $Mode \
    --root ../../data/emnlp \
    --encoder-class SPBPE \
    --vocab-size 30000;
