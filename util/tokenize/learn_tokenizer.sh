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
 --src-dir ../data/un_pc/raw \
 --target-dir ../data/un_pc/multi \
 --tokenizer-type WP \
 --prefix wp_30000 \
 --data-type un_pc \
 --use-control-token \
 --vocab-size 64000;