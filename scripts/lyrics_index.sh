#!/bin/bash

export PYTHONPATH="./"
echo $PYTHONPATH
python indexer/indexing.py \
    --directory_path  ../data/news\
    --file_name raw \
    --encoder_filename jamo-news-20000 \
    --df_filename jamo_indexed/news_indexed \
    --tokenizer_class mecab \
    --encoder_class sentencepiecebpe \
    --indexer_type news \
    --vocab_size 20000 \
    --split_jamo;
