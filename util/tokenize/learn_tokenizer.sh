#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

VOCAB_SIZE=30000

#python util/tokenize/main.py \
# --base-dir ../data/data-simplification/wikilarge/splited \
# --target-dir src \
# --tokenizer-type WP \
# --prefix wp_src_10000 \
# --data-type wikilarge \
# --vocab-size 10000;

python util/tokenize/main.py \
 --src-dir ../data/un_pc/raw/ar-en \
 --target-dir ../data/un_pc/mono/ar-en \
 --tokenizer-type WP \
 --prefix wp_$VOCAB_SIZE \
 --data-type un_pc \
 --vocab-size $VOCAB_SIZE;