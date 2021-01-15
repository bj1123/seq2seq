#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

#python util/tokenize/main.py \
# --base-dir ../data/data-simplification/wikilarge/splited \
# --target-dir src \
# --tokenizer-type WP \
# --prefix wp_src_10000 \
# --data-type wikilarge \
# --vocab-size 10000;

python util/tokenize/main.py \
 --base-dir ../data/multitask_access \
 --target-dir . \
 --tokenizer-type WP \
 --prefix wp_30000 \
 --data-type multitask \
 --vocab-size 30000;