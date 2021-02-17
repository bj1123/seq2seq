import torch
from torch import nn
from abc import ABC, abstractmethod
import math
from torch.nn import functional as F


class AttBase(nn.Module, ABC):
    def __init__(self, hidden_dim:int, n_head:int, head_dim:int,
                 dropout_rate:float, dropatt_rate:float=0.0, pre_lnorm=False):
        super(AttBase, self).__init__()
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.dropout_rate = dropout_rate

        self.kv_net = nn.Linear(hidden_dim, 2 * n_head * head_dim, bias=False)
        self.q_net = nn.Linear(hidden_dim, n_head * head_dim, bias=False)

        self.dropout = nn.Dropout(dropout_rate)
        self.dropatt = nn.Dropout(dropatt_rate)
        self.o_net = nn.Linear(n_head * head_dim, hidden_dim, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.scale = 1 / (head_dim ** 0.5)

        self.pre_lnorm = pre_lnorm

    @staticmethod
    def mask(score, mask):
        encoder_mask = mask.bool()
        score.masked_fill_(encoder_mask.unsqueeze(-1), -6e4)
        return score

    def attend(self, query, key, value, mask):
        bs, qs = query.size()[:2]
        score = self.compute_score(query, key)
        score = self.mask(score, mask)
        attended = self.linear_combine(score, value)

        out = self.o_net(attended.contiguous().view(bs, qs, -1))
        out = self.dropout(out)
        return out, score

    def compute_score(self, query, key):
        bs, qs, hs = query.size()
        ks = key.size(1)

        # print(query.size(),key.size(),value.size(),rel.size())
        # reshaping
        k = key.view(bs, ks, self.n_head, self.head_dim)
        q = query.view(bs, qs, self.n_head, self.head_dim)

        att_score = torch.einsum('bqnd,bknd->bqkn', q, k)
        att_score.mul_(self.scale)
        return att_score

    def linear_combine(self, score, value):
        bs, ks, hs = value.size()
        v = value.view(bs, ks, self.n_head, self.head_dim)
        att_prob = torch.softmax(score, 2)
        att_prob = self.dropatt(att_prob)
        attended = torch.einsum('bqkn,bknd->bqnd', att_prob, v)
        return attended

    def projection(self, q, kv, mem):
        kv = self.kv_net(kv)
        c = torch.cat([mem,kv],1)
        key, value = c.chunk(2,-1)
        query = self.q_net(q)
        return kv, query, key, value

    def before_add(self, q, kv, mem, mask):
        if kv is None:
            return q, None
        if mem is None:
            mem = torch.Tensor().to(device=kv.device, dtype=kv.dtype)

        if self.pre_lnorm:
            kv = self.layer_norm(kv)
            q = self.layer_norm(q)

        kv, query, key, value = self.projection(q, kv, mem)
        out, att_prob = self.attend(query, key, value, mask)
        return query, out, kv, att_prob


class MultiheadAtt(AttBase):
    def __init__(self, hidden_dim:int, n_head:int, head_dim:int,
                 dropout_rate:float, dropatt_rate:float=0.0, pre_lnorm=False, **kwargs):
        super(MultiheadAtt, self).__init__(hidden_dim, n_head, head_dim, dropout_rate, dropatt_rate, pre_lnorm)

    def forward(self, q, kv, mem, mask):
        query, out, kv, att_prob = self.before_add(q, kv, mem, mask)
        out = query + out
        if not self.pre_lnorm:
            out = self.layer_norm(out)
        return out, kv, att_prob


class SentenceAwareAtt(MultiheadAtt):
    def __init__(self, hidden_dim:int, n_head:int, head_dim:int,
                 dropout_rate:float, dropatt_rate:float=0.0, pre_lnorm=False):
        super(SentenceAwareAtt, self).__init__(hidden_dim, n_head, head_dim, dropout_rate, dropatt_rate, pre_lnorm)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, q, kv, mem, mask):
        query, out, kv, att_prob = self.before_add(q, kv, mem, mask)
        out = out - query
        out = self.proj(out)
        if not self.pre_lnorm:
            out = self.layer_norm(out)
        return out, kv, att_prob


class RelMultiheadAtt(AttBase):
    def __init__(self, hidden_dim:int, n_head:int, head_dim:int,
                 dropout_rate:float,
                 dropatt_rate:float=0.0, pre_lnorm=False, maxlen=512, relative_attention_num_buckets=128,
                 is_decoder=False):
        super(RelMultiheadAtt, self).__init__(hidden_dim, n_head, head_dim,
                                              dropout_rate, dropatt_rate, pre_lnorm)
        self.is_decoder = is_decoder
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.maxlen = maxlen
        self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_head)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """ Compute binned relative position bias """
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.maxlen
        )
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)[None]  # shape (query_length, key_length, num_heads)
        return values

    def compute_score(self, query, key):
        score = super().compute_score(query, key)
        ql = query.size(1)
        kl = key.size(1)
        position_bias = self.compute_bias(ql, kl)
        return score + position_bias

    def forward(self, q, kv, mem, mask):
        query, out, kv, att_prob = self.before_add(q, kv, mem, mask)
        out = query + out
        if not self.pre_lnorm:
            out = self.layer_norm(out)
        return out, kv, att_prob


class GraphRelMultiheadAtt(AttBase):
    def __init__(self, hidden_dim:int, n_head:int, head_dim:int,
                 dropout_rate:float,
                 dropatt_rate:float=0.0, pre_lnorm=False, maxlen=512, relative_attention_num_buckets=128,
                 is_decoder=False):
        super(GraphRelMultiheadAtt, self).__init__(hidden_dim, n_head, head_dim,
                                                   dropout_rate, dropatt_rate, pre_lnorm)
        self.is_decoder = is_decoder
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.maxlen = maxlen

        self.go_net = nn.Linear(hidden_dim, hidden_dim, bias=False)

    @staticmethod
    def _get_distance(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        relative_buckets = 0
        if bidirectional:
            relative_position = torch.abs(relative_position)
            masks = torch.full(relative_position.size(), False, device=relative_position.device)
        else:
            relative_position = -torch.min(relative_position, torch.ones_like(relative_position))
            masks = relative_position == -1
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        dist = 1 / (relative_buckets + 1)
        dist = dist.masked_fill(masks, 0)
        return dist

    def graph_att(self, q, k, graph_mem, mask):
        if graph_mem is None:
            graph_mem = torch.Tensor().to(device=k.device, dtype=k.dtype)
        graph_new_mem = k
        k = torch.cat([graph_mem,k],1)
        qs, ks = q.size(1), k.size(1)
        context_position = torch.arange(qs, dtype=torch.long, device=q.device)[:, None]
        memory_position = torch.arange(ks, dtype=torch.long, device=q.device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        dist = self._get_distance(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.maxlen
        )
        dist = dist[None].masked_fill(mask.bool(), 0).to(k.dtype)
        degree = dist.sum(-1)
        inv_degree = (1 / degree)[...,None]  # [bs, qs, 1]
        w = dist * inv_degree
        out = torch.matmul(w, k)
        return self.go_net(out), graph_new_mem

    def before_add(self, q, kv, mem, mask):
        if mem is None:
            graph_mem = plain_mem = None
        else:
            graph_mem, plain_mem = mem[..., -512:], mem[..., :-512]
        graph_out, graph_new_mem = self.graph_att(q, kv, graph_mem, mask)
        query, out, plain_new_mem, att_prob = super().before_add(q, kv, plain_mem, mask)  # from plain transformer
        new_mem = torch.cat([plain_new_mem,graph_new_mem], -1)
        return query, out, new_mem, att_prob, graph_out

    def forward(self, q, kv, mem, mask):
        query, out, kv, att_prob, graph_out = self.before_add(q, kv, mem, mask)
        out = query + out + graph_out
        if not self.pre_lnorm:
            out = self.layer_norm(out)
        return out, kv, att_prob
