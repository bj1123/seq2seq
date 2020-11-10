from model.transformer import *
from util.batch_generator import *
import json
from util.sampler import Sampler
import os
from util.args import SamplingArgument
from util.losses import *
import apex
from torch.utils.data.dataloader import DataLoader


def get_model(args):
    model = EncoderDecoderModel(args.vocab_size, args.batch_seqlen, args.hidden_dim, args.projection_dim, args.n_heads,
                                args.head_dim, args.n_enc_layers, args.n_dec_layers, args.dropout_rate,
                                args.dropatt_rate, args.padding_index, pre_lnorm=args.pre_lnorm,
                                rel_att=args.relative_pos)
    model.load_state_dict(torch.load(args.load_path))
    model = model.to(args.device)
    return model


def get_batchfier(args):
    test_batchfier = MTBatchfier(args.test_src_path, args.test_tgt_path, args.batch_size * 8, args.seq_len,
                                 padding_index=args.padding_index, epoch_shuffle=False,
                                 device=args.device, sampling_mode=True)
    return DataLoader(test_batchfier, args.batch_size * 8, collate_fn=test_batchfier.collate_fn)


def get_sampler(args, model, batchfier):
    def get_eos_index(batchfier):
        return batchfier.dataset.df['tgt_texts'][0][-1]

    optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)  # for mixed
    if args.mixed_precision:
        opt_level = 'O2'
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=opt_level)
    trainer = Sampler(model, args.sampling_mode, 200, args.temperature, args.width, get_eos_index(batchfier),
                      use_cache=True)
    return trainer


if __name__ == '__main__':
    args = SamplingArgument()
    # print(args.__dict__)
    model = get_model(args)
    batchfier = get_batchfier(args)
    sampler = get_sampler(args, model, batchfier)
    prev_step = 0
    res = []
    cnt = 0
    for inp in batchfier:
        cnt +=1
        print(cnt)
        res.extend(sampler.sample(inp))
    if not os.path.exists(os.path.dirname(args.sample_save_path)):
        os.makedirs(os.path.dirname(args.sample_save_path))
    json.dump(res, open(args.sample_save_path,'w'))
