#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

VOCAB_SIZE=140000

python util/tokenize/main.py \
 --src-dir ../data/game_mt/2ko \
 --target-dir ../data/game_mt/2ko \
 --tokenizer-type WP \
 --prefix wp_$VOCAB_SIZE \
 --data-type multilingual \
 --morph-analyzer-type none \
 --vocab-size $VOCAB_SIZE;

#python util/tokenize/main.py \
# --src-dir ../data/un_pc/raw/en-es \
# --target-dir ../data/un_pc/mono/en-es \
# --tokenizer-type WP \
# --prefix wp_$VOCAB_SIZE \
# --data-type un_pc \
# --vocab-size $VOCAB_SIZE;