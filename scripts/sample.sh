#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

Mode=unlikelihood-token
Data=bugs
for K in 5
do
  for S in 1.4
  do
    CUDA_VISIBLE_DEVICES=2 python lm_sample.py \
        --saved-path data/$Data/_$Mode''_layer_12_lr_0.0002_cutoffs_6_epoch_9 \
        --dataset $Data \
        --loss-type $Mode \
        --top-k $K \
        --temperature $S \
        --sampling-mode 2 \
        --root ../../data/emnlp \
        --encoder-class FP \
        --nprefix 50 \
        --ngenerate 100 \
        --vocab-size 30000;
  done
done
#
#for K in 1 3 10 20 40
#do
#  for S in 1.0 1.2 1.4
#  do
#    CUDA_VISIBLE_DEVICES=0 python lm_sample.py \
#        --saved-path data/$Data/_$Mode''_layer_12_lr_0.0002_cutoffs_6_epoch_9 \
#        --dataset $Data \
#        --loss-type $Mode \
#        --top-k $K \
#        --temperature $S \
#        --sampling-mode 1 \
#        --root ../../data/emnlp \
#        --encoder-class FP \
#        --nprefix 5 \
#        --ngenerate 300 \
#        --vocab-size 30000;
#  done
#done
#
#for K in 1 3 10 20 40
#do
#  for S in 1.0 1.2 1.4
#  do
#    CUDA_VISIBLE_DEVICES=0 python lm_sample.py \
#        --saved-path data/$Data/_$Mode''_layer_12_lr_0.0002_cutoffs_6_epoch_9 \
#        --dataset $Data \
#        --loss-type $Mode \
#        --top-k $K \
#        --temperature $S \
#        --sampling-mode 2 \
#        --root ../../data/emnlp \
#        --encoder-class FP \
#        --nprefix 5 \
#        --ngenerate 300 \
#        --vocab-size 30000;
#  done
#done