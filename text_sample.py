from model.transformer import *
from util.tokenize.data_specific_tokenizer import *
from torch.utils.data.dataloader import DataLoader
from util.args import get_args
from util.losses import *
from util.sampler import *
import apex
from util.batch_generator import *
from main import get_model


def get_batchfier(args):
    task = args.task
    if 'multitask' in task:
        tokenizer = MultiTaskTokenizer(args.dir_path, args.tokenizer_prefix)
        batchfier = MultitaskInferBatchfier(args.target_text_path, args.special_token_indice, tokenizer)
    else:
        raise NotImplementedError
    return tokenizer, DataLoader(batchfier, args.batch_size, collate_fn=batchfier.collate_fn)


def get_sampler(args, model, batchfier):
    optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)  # for mixed
    if args.mixed_precision:
        opt_level = 'O2'
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=opt_level)
    sampler = Sampler(model, args.sampling_mode, 200, args.temperature, args.width, batchfier.dataset.eos_idx,
                      use_cache=True, length_penalty=args.lengths_penalty)
    return sampler


def main(args):
    model = get_model(args)
    model.load_state_dict(torch.load(args.load_path))
    model = model.to(args.device)
    model.eval()
    tokenizer, dl = get_batchfier(args)
    sampler = get_sampler(args, model, dl)
    res = []
    for inp in dl:
        res.extend(sampler.sample(inp))
    for i in res:
        print(i)
        print(tokenizer.decode(i))


if __name__ == '__main__':
    args = get_args()
    main(args)