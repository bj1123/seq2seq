#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH


python util/tokenize/main.py \
 --directory-path ../data/multitask \
 --tokenizer-type WP \
 --data-type multitask \
 --multitask \
 --vocab-size 30000;