#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH


python util/metrics/sari.py \
--gt-path ../data/data-simplification/wikilarge/turkcorpus_raw_encoded_mapped \
--sample-path data/sampled/wikilarge/base_0.0001/epoch_4;