#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

python util/tokenize/main.py \
 --base-dir ../data/simplification \
 --target-dir wikilarge \
 --tokenizer-type WP \
 --data-type wikilarge \
 --vocab-size 10000;

#python util/tokenize/main.py \
# --directory-path ../data/multitask \
# --tokenizer-type WP \
# --data-type multitask \
# --multitask \
# --vocab-size 30000;