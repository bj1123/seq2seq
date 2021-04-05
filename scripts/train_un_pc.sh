#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

DATASET=un_pc/mono/en-es
MODEL=plain
SRC=es
TGT=en
PENCODING=absolute

CUDA_VISIBLE_DEVICES=0 python main.py \
 --src-path ../data/$DATASET/encoded_mapped/$SRC \
 --tgt-path ../data/$DATASET/encoded_mapped/$TGT \
 --loss-type plain \
 --positional-encoding $PENCODING \
 --task seq2seq \
 --model-type $MODEL \
 --dataset-name $DATASET \
 --model-size base;
