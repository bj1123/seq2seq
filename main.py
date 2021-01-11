from model.transformer import *
from util.batch_generator import *
from util.files import *
from util.initializer import *
from util.trainer import Trainer
import os
from util.args import *
from util.losses import *
import apex
from util.lr_scheduler import *


def get_model(args):
    print(args.vocab_size, args.batch_seqlen, args.hidden_dim, args.projection_dim, args.n_heads,
          args.head_dim, args.n_enc_layers, args.n_dec_layers, args.dropout_rate,
          args.dropatt_rate, args.padding_index, args.shared_embedding, args.tie_embedding)
    if args.task in ('seq2seq', 'access'):
        if args.model_type == 'complexity-aware':
            model = ComplexityAwareModel(args.vocab_size, args.batch_seqlen, args.hidden_dim, args.projection_dim,
                                         args.n_heads, args.head_dim, args.n_enc_layers, args.n_dec_layers,
                                         args.dropout_rate,
                                         args.dropatt_rate, args.padding_index, args.cutoffs, pre_lnorm=args.pre_lnorm,
                                         rel_att=args.relative_pos, shared_embedding=args.shared_embedding,
                                         tie_embedding=args.tie_embedding)

        elif args.model_type == 'sentence-aware':
            model = SentenceAwareModel(args.vocab_size, args.batch_seqlen, args.hidden_dim, args.projection_dim,
                                       args.n_heads, args.head_dim, args.n_enc_layers, args.n_dec_layers,
                                       args.dropout_rate,
                                       args.dropatt_rate, args.padding_index, pre_lnorm=args.pre_lnorm,
                                       rel_att=args.relative_pos, shared_embedding=args.shared_embedding,
                                       tie_embedding=args.tie_embedding)

        else:
            model = EncoderDecoderModel(args.vocab_size, args.batch_seqlen, args.hidden_dim, args.projection_dim,
                                        args.n_heads, args.head_dim, args.n_enc_layers, args.n_dec_layers,
                                        args.dropout_rate,
                                        args.dropatt_rate, args.padding_index, pre_lnorm=args.pre_lnorm,
                                        rel_att=args.relative_pos, shared_embedding=args.shared_embedding,
                                        tie_embedding=args.tie_embedding)
    elif args.task == 'multitask':
        model = CrossLingualModel(args.vocab_size, args.batch_seqlen, args.hidden_dim, args.projection_dim,
                                  args.n_heads, args.head_dim, args.n_enc_layers, args.n_dec_layers,
                                  args.dropout_rate, args.dropatt_rate, args.padding_index, pre_lnorm=args.pre_lnorm,
                                  rel_att=args.relative_pos, shared_embedding=args.shared_embedding,
                                  tie_embedding=args.tie_embedding)
    initializer = Initializer('normal', 0.02, 0.1)
    initializer.initialize(model)
    model = model.to(args.device)
    return model


def get_batchfier(args):
    if args.task == 'seq2seq':
        if args.model_type == 'complexity-aware':
            train_batchfier = ComplexityControlBatchfier(args.train_src_path, args.train_tgt_path, args.rare_index,
                                                         args.batch_size, args.seq_len,
                                                         padding_index=args.padding_index, device=args.device)
            test_batchfier = ComplexityControlBatchfier(args.test_src_path, args.test_tgt_path, args.rare_index,
                                                        args.batch_size, args.seq_len,
                                                        padding_index=args.padding_index, device=args.device)
        else:
            train_batchfier = MTBatchfier(args.train_src_path, args.train_tgt_path, args.batch_size, args.seq_len,
                                          padding_index=args.padding_index, device=args.device)
            test_batchfier = MTBatchfier(args.test_src_path, args.test_tgt_path, args.batch_size, args.seq_len,
                                         padding_index=args.padding_index, device=args.device)
    elif args.task == 'multitask':
        train_batchfier = MultitaskBatchfier(args.train_path, args.special_token_indice, args.batch_size, args.seq_len,
                                             padding_index=args.padding_index, device=args.device)
        test_batchfier = MultitaskBatchfier(args.test_path, args.special_token_indice, args.batch_size, args.seq_len,
                                            padding_index=args.padding_index, device=args.device)
    elif args.task == 'access':
        train_batchfier, test_batchfier = get_fair_batchfier(args.dataset, args.device)
    else:
        raise NotImplementedError
    return train_batchfier, test_batchfier


def get_loss(args, train_batchfier):
    lt = args.loss_type
    print(lt)
    if lt == 'plain':
        if args.label_smoothing > 0:
            loss = LabelSmoothingLoss(args.vocab_size, ignore_index=train_batchfier.padding_index, device=args.device)
        else:
            loss = PlainLoss(train_batchfier.padding_index)
    else:
        raise NotImplementedError
    if args.model_type == 'complexity-aware':
        loss = ComplexityLoss(loss)
    elif args.model_type == 'sentence-aware':
        loss = SentenceAwareLoss(loss)
    return loss


def get_trainer(args, model, train_batchfier, test_batchfier):
    optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate,
                                  weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    if args.mixed_precision:
        print('mixed_precision')
        opt_level = 'O2'
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=opt_level)
    decay_step = len(train_batchfier) * args.n_epoch // args.update_step
    scheduler = WarmupLinearSchedule(optimizer, args.warmup_step, decay_step, args.decay_on_valid)
    # scheduler = WarmupExponentialSchedule(optimizer, args.warmup_step, len(train_batchfier) // args.update_step)
    criteria = get_loss(args, train_batchfier)
    trainer = Trainer(model, train_batchfier, test_batchfier, optimizer, scheduler, args.update_step, criteria,
                      args.clip_norm, args.mixed_precision, save_name=args.savename,
                      n_ckpt=5)
    return trainer


if __name__ == '__main__':
    args = get_args()
    # print(args.__dict__)
    model = get_model(args)
    train_batchfier, test_batchfier = get_batchfier(args)
    trainer = get_trainer(args, model, train_batchfier, test_batchfier)
    prev_step = 0
    res = []

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('# params : {}'.format(params))
    if not os.path.exists(args.savename):
        os.makedirs(args.savename)

    for i in range(args.n_epoch):
        print('epoch {}'.format(i + 1))
        trainer.train_epoch()
        test_loss = trainer.test_epoch()
        res.append(test_loss)
        savepath = os.path.join(args.savename, 'epoch_{}'.format(i))
        torch.save(model.state_dict(), savepath)
        # test
    print(res)
