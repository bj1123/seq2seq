from model.transformer import *
from util.batch_generator import *
import json
from util.sampler import *
import os
from util.args import get_args
from util.losses import *
import apex
from torch.utils.data.dataloader import DataLoader
import time
from main import get_model


def get_batchfier(args):
    test_batchfier = MTBatchfier(args.test_src_path, args.test_tgt_path, args.batch_size, args.seq_len,
                                 padding_index=args.padding_index, epoch_shuffle=False,
                                 device=args.device, sampling_mode=True)
    return DataLoader(test_batchfier, args.batch_size, collate_fn=test_batchfier.collate_fn)


def get_sampler(args, model, batchfier):
    optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)  # for mixed
    if args.mixed_precision:
        opt_level = 'O2'
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=opt_level)

    if args.complexity_aware:
        trainer = ComplexitySampler(model, args.sampling_mode, 200, args.temperature, args.width,
                                    batchfier.dataset.eos_idx, use_cache=True, length_penalty=args.lengths_penalty)

    else:
        trainer = Sampler(model, args.sampling_mode, 200, args.temperature, args.width, batchfier.dataset.eos_idx,
                          use_cache=True, length_penalty=args.lengths_penalty)
    return trainer


if __name__ == '__main__':
    args = get_args()
    # print(args.__dict__)
    model = get_model(args)
    model.load_state_dict(torch.load(args.load_path))
    model = model.to(args.device)
    model.eval()
    batchfier = get_batchfier(args)
    sampler = get_sampler(args, model, batchfier)
    prev_step = 0
    res = []
    cnt = 0
    t = time.time()
    for inp in batchfier:
        cnt +=1
        print(cnt)
        res.extend(sampler.sample(inp))
    if not os.path.exists(os.path.dirname(args.sample_save_path)):
        os.makedirs(os.path.dirname(args.sample_save_path))
    print(time.time()-t)
    # f = open(args.sample_save_path, 'w')
    # txts = [' '.join(map(str, i[:-1])) + ' \n' for i in res]
    # f.writelines(txts)
    json.dump(list(map(lambda x: x[:-1],res)), open(args.sample_save_path,'w'))
