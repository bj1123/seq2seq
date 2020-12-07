from util.tokenize.data_specific_tokenizer import *
from util.tokenize.base_tokenizer import *
from util.tokenize.data_reformatter import *

MORPHS_ANALYZER_MAP = {'mecab': MecabAnalyzer, 'none': NullAnalyzer}
TOKENIZER_MAP = {'sentencepiecebpe': tokenizers.SentencePieceBPETokenizer,
                 'wordpiece': tokenizers.BertWordPieceTokenizer}
DATA_MAP = {'mtspace': MTSpaceTokenizer, 'multitask': XLSXMultiTaskTokenizer}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory-path", type=str,
                        help='parent directory path')
    parser.add_argument("--inp-path", type=str,
                        help='directory where input data is stored')
    parser.add_argument("--encoder-filename", type=str, default=r"mt",
                        help='encoder will be stored with this name')
    parser.add_argument("--out-path", type=str,
                        help='directory path where encoded data is stored')
    parser.add_argument("--tokenizer-class", type=str, default="SP")
    parser.add_argument("--data-type", type=str, default="SP")
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--morph-analyzer-class", type=str, default="none")
    parser.add_argument("--split-jamo", action='store_true')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    morphs_analyzer_class = MORPHS_ANALYZER_MAP[args.tokenizer_class]
    tokenizer_class = TOKENIZER_MAP[args.subencoder_class]
    data_class = DATA_MAP[args.indexer_type]
    indexer = data_class(args.encoder_class, args.use_morphs, vocab_size=args.vocab_size,
                         tokenizer_class=tokenizer_class, morph_analyzer_class=morphs_analyzer_class,
                         jamo=args.split_jamo)
    print(args.directory_path, args.inp_path)
    indexer.corpus_encode(args.inp_path)


if __name__ == "__main__":
    main()
