#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

PENCODING=absolute
MODEL=plain
EPOCH=10

python evaluate.py \
--gt-path ../data/wmt/de_encoded/test.pkl \
--sample-path data/sampled/wmt/$MODEL/base_0.00021''_$PENCODING/epoch_$EPOCH

perl multi-bleu.perl gt.txt < sampled.txt