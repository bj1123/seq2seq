from model.transformer import *
from util.batch_generator import *
import json
from util.sampler import *
import os
from util.args import get_args
from util.losses import *
import apex
import time
from main import get_model


def get_batchfier(args):
    if args.task == 'access':
        test_batchfier = FairTestBatchfier(args.dataset, args.batch_size * 4, device=args.device)

    elif args.task =='seq2seq':
        # test_batchfier = TorchTextMT(args.train_src_path, args.train_tgt_path, args.batch_size // args.update_step,
        #                               padding_index=args.padding_index, device=args.device,
        #                               epoch_shuffle=False, sampling_mode=True)
        test_batchfier = MTBatchfier(args.test_src_path, args.test_tgt_path, 64, seq_len=args.seq_len,
                                     padding_index=args.padding_index, epoch_shuffle=False,
                                     device=args.device, sampling_mode=True)
    return test_batchfier.to_iterator()


def get_sampler(args, model, batchfier):
    optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)  # for mixed
    if args.mixed_precision:
        opt_level = 'O2'
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=opt_level)

    if args.model_type == 'complexity-aware':
        trainer = ComplexitySampler(model, args.sampling_mode, 400, args.temperature, args.width,
                                    batchfier.dataset.eos_idx, use_cache=True, length_penalty=args.lengths_penalty)
    elif args.model_type == 'sentence-aware':
        trainer = SentenceAwareSampler(model, args.sampling_mode, 400, args.temperature, args.width,
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
        res.extend(sampler.sample(inp))
        # print(res)
        print(cnt, time.time()-t)
        t = time.time()
    if not os.path.exists(os.path.dirname(args.sample_save_path)):
        os.makedirs(os.path.dirname(args.sample_save_path))
    # f = open(args.sample_save_path, 'w')
    # txts = [' '.join(map(str, i[:-1])) + ' \n' for i in res]
    # f.writelines(txts)
    json.dump(list(map(lambda x: x[:-1],res)), open(args.sample_save_path,'w'))
