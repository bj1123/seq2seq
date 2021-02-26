#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

PENCODING=absolute
MODEL=sentence-aware

python evaluate.py \
--gt-path ../data/wmt/de_encoded/test.pkl \
--sample-path data/sampled/wmt/$MODEL/base_0.00021''_$PENCODING/epoch_10

perl multi-bleu.perl gt.txt < sampled.txt