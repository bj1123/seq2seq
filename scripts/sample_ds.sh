#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH


CUDA_VISIBLE_DEVICES=0 python sample.py \
  --src-path ../data/wmt/en_encoded \
  --tgt-path ../data/wmt/de_encoded \
  --saved-model-folder data/saved_model/base_0.00025 \
  --saved-model-ckpt epoch_9 \
  --sample-save-path data/sampled/base_0.00025/epoch_9_averaged \
  --sampling-mode beam \
  --width 4 \
  --lengths-penalty 0.6 \
  --model-size base;
