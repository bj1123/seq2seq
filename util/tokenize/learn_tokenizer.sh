#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

VOCAB_SIZE=30000

python util/tokenize/main.py \
 --src-dir ../data/game_mt/raw/ko2en/ko \
 --target-dir ../data/game_mt/raw/ko2en/splited \
 --tokenizer-type WP \
 --prefix wp_ko_$VOCAB_SIZE \
 --data-type wikilarge \
 --morph-analyzer-type mecab \
 --vocab-size $VOCAB_SIZE;

#python util/tokenize/main.py \
# --src-dir ../data/un_pc/raw/en-es \
# --target-dir ../data/un_pc/mono/en-es \
# --tokenizer-type WP \
# --prefix wp_$VOCAB_SIZE \
# --data-type un_pc \
# --vocab-size $VOCAB_SIZE;