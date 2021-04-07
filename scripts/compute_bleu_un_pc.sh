#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

PENCODING=absolute
MODEL=plain
EPOCH=8

python evaluate.py \
--gt-path ../data/un_pc/multi/encoded_mapped/ar-en/en/test.feather \
--sample-path data/sampled/un_pc/multi/$MODEL/base_0.00021''_$PENCODING/epoch_$EPOCH

perl multi-bleu.perl gt.txt < sampled.txt


#--gt-path ../data/un_pc/mono/ar-en/encoded_mapped/en/test.feather \
#--sample-path data/sampled/un_pc/mono/ar-en/$MODEL/base_0.00021''_$PENCODING/epoch_$EPOCH
