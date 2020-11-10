#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH


CUDA_VISIBLE_DEVICES=0 python sample.py \
  --src-path ../data/wmt/en_encoded \
  --tgt-path ../data/wmt/de_encoded \
  --saved-model-folder data/saved_model/base_0.001 \
  --saved-model-ckpt epoch_0 \
  --sample-save-path data/sampled/base_0.001/epoch_0 \
  --sampling-mode beam \
  --width 6 \
  --model-size base;
