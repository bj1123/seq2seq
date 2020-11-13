from model.ops import mask_lengths
import re
import torch
import numpy as np
from copy import deepcopy


def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits
    else:
        values, _ = torch.topk(logits, k=k)
        min_values = values[:, -1, None]
        return torch.where(
            logits < min_values,
            torch.ones_like(logits, dtype=logits.dtype) * -1e4,
            logits,
        )


def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch = logits.size(0)
    sorted_logits, _ = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    a = torch.arange(0, batch).to(logits.device)
    b = torch.max(torch.sum(cumulative_probs <= p, dim=-1) - 1, torch.Tensor([0]).long().to(logits.device))
    min_values = sorted_logits[a, b].to(logits.device)
    return torch.where(
        logits < min_values[:, None],
        torch.ones_like(logits) * -1e4,
        logits,
    )


def block_words(generated, ngram):
    target = ' '.join(map(str, generated[-ngram + 1:]))
    temp = ' '.join(map(str, generated))
    blocked = re.findall('(?<={} )\d+'.format(target), temp)
    return [int(i) for i in blocked]


class Sampler:
    def __init__(self, model, mode, max_len, temperature, width, eos_index, use_cache=True, **kwargs):
        mode = mode.replace('_','-')
        assert mode in ['top-k', 'top-p', 'beam']
        self.model = model
        self.mode = mode
        self.max_len = max_len
        self.temperature = temperature
        self.width = width  # top-k if mode == top-k beam width if mode == beam and top-p probs.
        self.block = kwargs.pop('block', None)  # block repetition
        self.use_cache = use_cache
        self.eos_index = eos_index
        self.sample_fn = self.stochastic_sample if mode in ['top-k', 'top-p'] else self.beam_sample
        self.lp_a = kwargs.pop('length_penalty', None)

    def _length_normalize(self, probs, generated_texts):
        if self.lp_a:
            l = self._get_lengths(generated_texts)  # [bs, width]
            lp = (5 + l)**self.lp_a / 6**self.lp_a
            return probs / lp[...,None]
        else:
            return probs

    def _get_lengths(self, generated_texts):
        """
        return the length of generated texts till the eos token
        :param generated_texts: torch.LongTensor size = (bs, width, len)
        :return:
        """
        eos_index = self.eos_index
        bs, width, lengths = generated_texts.size()
        ind = (generated_texts == eos_index).nonzero(as_tuple=True)
        x = torch.full((bs, width), lengths + 1, dtype=torch.float32).to(generated_texts.device)
        x[ind[0].flip(0), ind[1].flip(0)] = ind[-1].flip(0).to(x.dtype) + 1
        return x

    def get_maxlen(self, inp):
        batch_len = inp['src'].size(-1)
        return min(self.max_len, int(batch_len * 1.2))

    def truncate(self, indexed):
        if self.eos_index not in indexed:
            return indexed
        else:
            return indexed[:indexed.index(self.eos_index)+1]

    def _mask_probs(self, probs, generated_texts):
        eos_index = self.eos_index
        mask = (generated_texts == eos_index).sum(-1) > 0
        return probs.masked_fill(mask[...,None], 0)

    def _beam_start(self, out, inp):
        logits = out['logits'][:,-1]
        bs = out['logits'].size(0)
        logits = logits / self.temperature
        probs, i = torch.topk(torch.log_softmax(logits, -1), self.width, -1)  # [batch, beam_size]
        if self.use_cache:
            new_tgt = i.view(-1,1)
            tgt_mem = out['tgt_mem']
            s = tgt_mem[0].size()
            tgt_mem = [i[:, None].repeat(1, self.width, 1, 1).view(-1, *s[1:]) for i in out['tgt_mem']]
            inp['tgt_mem'] = tgt_mem
        else:
            new_tgt = inp['tgt']
            new_tgt = torch.cat([new_tgt[:, None].repeat(1, self.width, 1), i[..., None]], 2) \
                .view(bs * self.width, -1)  # [batch, beam, l]
            inp['tgt_len'] = inp['tgt_len'] + 1
        inp['tgt'] = new_tgt
        size = inp['enc_out'].size()
        inp['enc_out'] = inp['enc_out'][:, None].repeat(1, self.width, 1, 1).view(-1, *size[1:])
        inp['src_len'] = inp['src_len'][:, None].repeat(1,self.width).view(-1)
        inp['tgt_len'] = inp['tgt_len'][:, None].repeat(1,self.width).view(-1)
        inp['history'] = i[..., None]  # [bs, width, 1]
        return probs, inp

    def _beam_continue(self, out, probs, inp):
        logits = out['logits'][:,-1]
        bs = logits.size(0) // self.width
        logits = logits / self.temperature
        logits = logits.view(bs, self.width, -1)
        # logits = block(logits,res,block_ngram)
        p, i = torch.topk(torch.log_softmax(logits, -1), self.width, -1)  # [batch_size, beam_size, beam_size]
        p = self._mask_probs(p, inp['history'])
        probs = probs.unsqueeze(-1) + p
        _, ni = self._length_normalize(probs, inp['history']).view(bs, -1).topk(self.width, -1)
        probs = probs.view(bs,-1).gather(1,ni)
        sampled = i.view(bs, -1).gather(1, ni)  # [bs, width]
        group = ni // self.width  # [bs, width]
        ind = [torch.arange(bs)[:, None], group]
        selected = inp['tgt'].view(bs, self.width, -1)[ind]
        if self.use_cache:
            new_tgt = sampled.view(-1,1)
            tgt_mem = inp['tgt_mem']  # [bs * width, l, hidden]
            new_mem = out['tgt_mem']  # [bs * width, l, hidden]
            hidden_size = tgt_mem[0].size(-1)
            new_tgt_mem = [torch.cat(
                [tgt_mem[i].view(bs, self.width, -1, hidden_size)[ind],
                 new_mem[i].view(bs, self.width, -1, hidden_size)[ind]], 2).view(bs*self.width,-1,hidden_size)
             for i in range(len(tgt_mem))]
            inp['tgt_mem'] = new_tgt_mem
        else:
            new_tgt = torch.cat([selected, sampled[..., None]], 2).view(bs * self.width, -1)
            inp['tgt_len'] = inp['tgt_len'] + 1
        inp['tgt'] = new_tgt
        inp['history'] = torch.cat([inp['history'][ind], sampled[..., None]], 2)
        return probs, inp

    def _beam_finalize(self, probs, inp):
        bs = inp['tgt'].size(0) // self.width
        _, ind = probs.topk(1, -1)  # [bs, width]
        return inp['history'][torch.arange(bs), ind.squeeze(-1)]

    def stochastic_sample(self, inp):
        def _merge_mem(inp, out):
            tgt_mem = inp['tgt_mem'] if 'tgt_mem' in inp else None
            new_mem = out['tgt_mem']
            if tgt_mem:
                inp['tgt_mem'] = [torch.cat([tgt_mem[i], new_mem[i].to(tgt_mem[i].dtype)], 1)
                                  for i in range(len(tgt_mem))]
            else:
                inp['tgt_mem'] = new_mem
            return inp

        def _merge_input(inp, sampled):
            if self.use_cache:
                inp['tgt'] = sampled
            else:
                inp['tgt'] = torch.cat([inp['tgt'], sampled], 1)
                inp['tgt_len'] = inp['tgt_len'] + 1
            return inp

        if self.mode == 'top-k':
            top_sth = top_k_logits
        elif self.mode == 'top-p':
            top_sth = top_p_logits
        else:
            raise NotImplementedError('only support top-k or top-p sampling for stochastic sampling')
        max_len = self.get_maxlen(inp)
        res = torch.LongTensor([]).to(inp['src'].device)
        with torch.no_grad():
            enc_out = self.model.encode_src(inp)
            inp['enc_out'] = enc_out
            for _ in range(max_len):
                out = self.model(inp)
                logits = top_sth(out['logits'][:,-1], self.width)
                logits = logits / self.temperature
                sampled = torch.multinomial(torch.softmax(logits, -1), 1)
                res = torch.cat([res, sampled], 1)
                inp = _merge_input(inp, sampled)
                if self.use_cache:
                    inp = _merge_mem(inp, out)
        return res

    def beam_sample(self, inp):
        probs = torch.zeros((inp['src'].size(0), self.width), dtype=inp['src'].dtype, device=inp['src'].device)
        cnt = 0
        max_len = self.get_maxlen(inp)
        with torch.no_grad():
            enc_out = self.model.encode_src(inp)
            inp['enc_out'] = enc_out
            for _ in range(max_len):
                cnt += 1
                out = self.model(inp)
                if cnt == 1:
                    probs, inp = self._beam_start(out, inp)
                else:
                    probs, inp = self._beam_continue(out, probs, inp)
        res = self._beam_finalize(probs, inp)
        return res

    def sample(self, inp):
        sampled = self.sample_fn(inp).tolist()
        return [self.truncate(i) for i in sampled]




