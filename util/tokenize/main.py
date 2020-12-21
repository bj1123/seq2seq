from util.tokenize.data_specific_tokenizer import *
from util.tokenize.base_tokenizer import *
from util.tokenize.data_reformatter import *

MORPHS_ANALYZER_MAP = {'mecab': MecabAnalyzer, 'none': NullAnalyzer}
TOKENIZER_CLASS_MAP = {'sentencepiecebpe': 'bpe',
                 'SP': 'bpe',
                 'wordpiece': 'wp',
                 'WP': 'wp',
                 }
TOKENIZER_MAP = {'mtspace': MTSpaceTokenizer, 'multitask': MultiTaskTokenizer, 'wikilarge':HFBaseTokenizer}
REFORMATTER_MAP = {'multitask':MultitaskReformatter, 'wikilarge':DSReformatter}

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str,
                        help='parent directory path')
    parser.add_argument("--target-dir", type=str,
                        help='parent directory path')
    parser.add_argument("--tokenizer-type", type=str, default="SP")
    parser.add_argument("--data-type", type=str, default="multitask")
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--morph-analyzer-type", type=str, default="none")
    parser.add_argument("--split-jamo", action='store_true')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    morphs_analyzer_class = MORPHS_ANALYZER_MAP[args.morph_analyzer_type]
    tokenizer_class = TOKENIZER_CLASS_MAP[args.tokenizer_type]
    reformatter_class = REFORMATTER_MAP[args.data_type]
    tokenizer = TOKENIZER_MAP[args.data_type]
    tokens_to_add = None
    reformatter = reformatter_class(os.path.join(args.base_dir,args.target_dir))
    reformatter.start()
    tokens_to_add = reformatter.tokens_to_add if hasattr(reformatter,'tokens_to_add') else None

    prefix = f'{args.tokenizer_type}_{args.vocab_size}'
    indexer = tokenizer(args.base_dir, prefix, vocab_size=args.vocab_size,
                         tokenizer_class=tokenizer_class, morph_analyzer_class=morphs_analyzer_class,
                         tokens_to_add = tokens_to_add,
                         jamo=args.split_jamo)
    indexer.corpus_encode(args.target_dir)


if __name__ == "__main__":
    main()
