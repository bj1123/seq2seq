from util.tokenize.data_specific_tokenizer import *
from util.tokenize.base_tokenizer import *
from util.tokenize.data_reformatter import *

MORPHS_ANALYZER_MAP = {'mecab': MecabAnalyzer, 'none': NullAnalyzer}
TOKENIZER_MAP = {'sentencepiecebpe': tokenizers.SentencePieceBPETokenizer,
                 'SP': tokenizers.SentencePieceBPETokenizer,
                 'wordpiece': tokenizers.BertWordPieceTokenizer,
                 'WP': tokenizers.BertWordPieceTokenizer,
                 }
DATA_MAP = {'mtspace': MTSpaceTokenizer, 'multitask': XLSXMultiTaskTokenizer}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory-path", type=str,
                        help='parent directory path')
    parser.add_argument("--tokenizer-type", type=str, default="SP")
    parser.add_argument("--data-type", type=str, default="multitask")
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--morph-analyzer-type", type=str, default="none")
    parser.add_argument("--multitask", action='store_true')
    parser.add_argument("--split-jamo", action='store_true')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    morphs_analyzer_class = MORPHS_ANALYZER_MAP[args.morph_analyzer_type]
    tokenizer_class = TOKENIZER_MAP[args.tokenizer_type]
    data_class = DATA_MAP[args.data_type]
    tokens_to_add = None
    if args.multitask:
        reformatter = MultitaskReformatter(args.directory_path)
        reformatter.start()
        tokens_to_add = reformatter.tokens_to_add
    prefix = f'{args.tokenizer_type}_{args.vocab_size}'
    indexer = data_class(args.directory_path, prefix, vocab_size=args.vocab_size,
                         tokenizer_class=tokenizer_class, morph_analyzer_class=morphs_analyzer_class,
                         tokens_to_add = tokens_to_add,
                         jamo=args.split_jamo)
    print(f' directory path: {args.directory_path}')
    indexer.corpus_encode('.')


if __name__ == "__main__":
    main()
