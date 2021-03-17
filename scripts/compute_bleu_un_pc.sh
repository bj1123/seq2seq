#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

PENCODING=absolute
MODEL=plain
EPOCH=1

python evaluate.py \
--gt-path ../data/un_pc/multi/encoded_mapped/ar-en/en/test.feather \
--sample-path data/sampled/un_pc/multi/ar-en/$MODEL/base_0.00021''_$PENCODING/epoch_$EPOCH

perl multi-bleu.perl gt.txt < sampled.txt