#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH


python evaluate.py --gt-path ../data/wmt/de_encoded/test.pkl \
  --sample-path data/sampled/base_0.00025/epoch_4;
