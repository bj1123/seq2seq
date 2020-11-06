!/bin/bash

export PYTHONPATH="./"
echo $PYTHONPATH
python indexer/indexing.py \
    --directory_path  ../data/poets\
    --file_name new_poetry3.csv \
    --encoder_filename news-20000 \
    --df_filename indexed/poet_indexed.pkl \
    --tokenizer_class mecab \
    --encoder_class sentencepiecebpe \
    --indexer_type poet \
    --vocab_size 20000
