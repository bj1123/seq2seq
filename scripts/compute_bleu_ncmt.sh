#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

PENCODING=absolute
MODEL=plain
EPOCH=8

python evaluate.py \
--gt-path ../data/game_mt/raw/ja2ko/splited/ko/encoded_mapped/test.ko.feather \
--sample-path data/sampled/game_mt/raw/ja2ko/$MODEL/base_0.00021''_$PENCODING/epoch_$EPOCH
#--sample-path zh

#python evaluate.py \
#--gt-path ../data/game_mt/2ko/encoded_mapped/zh2en/en/test.en.feather \
#--sample-path data/sampled/game_mt/semi/en/$MODEL/_base_0.00021''_$PENCODING/epoch_$EPOCH''_zh

perl multi-bleu.perl gt.txt < sampled.txt

#python evaluate.py \
#--gt-path ../data/game_mt/raw/zh2en/splited/en/encoded_mapped/test.en.feather \
#--sample-path data/sampled/game_mt/raw/zh2en/$MODEL/base_0.00021''_$PENCODING/epoch_$EPOCH


#--gt-path ../data/un_pc/mono/ar-en/encoded_mapped/en/test.feather \
#--sample-path data/sampled/un_pc/mono/ar-en/$MODEL/base_0.00021''_$PENCODING/epoch_$EPOCH
