#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH


CUDA_VISIBLE_DEVICES=0 python main.py \
--task access \
--dataset _2912c535c2343258d2e6375bca3e3a3d \
--loss-type plain \
--model-size base;
