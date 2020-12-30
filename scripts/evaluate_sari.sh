#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH


python util/metrics/sari.py \
--sample-path data/sampled/wikilarge/complexity/base_0.00011/epoch_8 \
--decoder-dir ../data/data-simplification/wikilarge/splited \
--prob-path ../data/data-simplification/wikilarge/splited/wp_tgt_10000-probs.json \
--decoder-prefix wp_tgt_10000;