import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.ops import *
from typing import Dict, Optional, Tuple, Any, List
import time

class Word_Embedding(nn.Module):
    def __init__(self,morph_size:int, pos_size:int, morph_lstm_hidden_size:int, morph_embedding_size:int,
                 pos_embedding_size, cutoffs:list, div_val:int, dropout:float=0.0, adaptive_embedding=False):
        super(Word_Embedding, self).__init__()

        self.morph_size = morph_size
        self.pos_size = pos_size
        self.morph_embedding_size = morph_embedding_size
        self.tag_embedding_size = pos_embedding_size
        self.morph_hidden_size = morph_lstm_hidden_size
        if not adaptive_embedding:
            self.morph = nn.Embedding(morph_size+1,morph_embedding_size,morph_size)
        else:
            self.morph = AdaptiveEmbedding(morph_size+1,morph_embedding_size,morph_embedding_size,cutoffs,div_val)

        self.pos = nn.Embedding(pos_size+1,pos_embedding_size,pos_size)
        self.padding = nn.ConstantPad1d(3,0.0)
        self.Ks = [2,3,4]

        self.embedding_size = morph_lstm_hidden_size
        # self.embedding_size = len(self.Ks) * filter_size + self.morph_embedding_size + self.tag_embedding_size
        self.morph_rnn = nn.LSTM(morph_embedding_size+pos_embedding_size, morph_lstm_hidden_size, 1,
                                 batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def morph_embedding(self,morphs,tags, morphs_lens):
        """

        :param morphs: torch.Tensor size of [batch, sentence_length, word_length]
        :param tags: torch.Tensor size of [batch, sentence_length, word_length]
        :param morphs_lens: torch.Tensor size of [batch, sentence_length]
        :return: torch.Tensor size of [batch,sentence_lengths,embedding_size]
        """
        b,s,w = morphs.size()
        # print(morphs.size(),tags.size(),morphs_lens.size())
        m = self.morph(morphs)
        t = self.pos(tags)
        c = torch.cat([m,t],-1) #[batch,sentence_lengths,word_lengths,embedding]
        # c = c.sum(-2)
        c = c.view(b*s,w,-1)
        morphs_lens = morphs_lens.view(b*s)
        lens_mask = mask_lengths(morphs_lens)
        zero_up_morphs_lens = torch.max(morphs_lens,torch.ones_like(morphs_lens,dtype=morphs_lens.dtype))
        rnned = run_rnn(c,zero_up_morphs_lens,self.morph_rnn)
        rnned = rnned * lens_mask.to(rnned.dtype).unsqueeze(-1)

        pooled = last_pool(rnned,zero_up_morphs_lens)
        outs = pooled.view(b,s,-1)

        return outs, rnned.view(b,s,w,-1)
        # return c

    def forward(self,morphs, pos, morphs_lens):
        mres = self.morph_embedding(morphs, pos, morphs_lens)
        # cres = self.character_embedding(characters,characters_lens)
        # res = torch.cat([mres, cres],-1)
        return mres


class StructuredEmbedding(nn.Module):
    def __init__(self, vocab_size:int, embedding_dim:int, n_cluster=None, entry_per_cluster=None, base_embedding=50):
        super(StructuredEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_dim
        self.n_cluster = n_cluster if n_cluster else 5
        self.entry_per_cluster = entry_per_cluster if entry_per_cluster else 1000 // self.n_cluster
        self.cluster_weights = nn.Embedding(vocab_size, self.n_cluster)
        self.base_weights = nn.Embedding(vocab_size, base_embedding)
        self.base_embeddings = nn.ParameterList([nn.Parameter(torch.Tensor(self.entry_per_cluster, embedding_dim))
                                                 for _ in range(self.n_cluster)])
        for i in self.base_embeddings:
            torch.nn.init.normal_(i, std=0.02)
        self.proj = nn.Sequential(
            nn.Linear(base_embedding, self.n_cluster * self.entry_per_cluster), nn.ReLU(),
            nn.Linear(self.n_cluster * self.entry_per_cluster, self.n_cluster * self.entry_per_cluster)
        )
        self.ln = nn.LayerNorm(embedding_dim)
        self.scale = nn.Parameter(torch.Tensor([0.02]))

    def forward(self, x):
        cluster_weights = self.cluster_weights(x)  # [bs, l, n_clusters]
        tgt_embs = []
        base_weights = self.proj(self.base_weights(x))
        for i in range(self.n_cluster):
            l, r = i * self.entry_per_cluster, (i+1) * self.entry_per_cluster
            target_base_weights = base_weights[...,l:r]  # [bs, l, entry]
            tgt_embs.append(torch.matmul(target_base_weights, self.base_embeddings[i]))  # [bs, l, emb]
        tgt_embs = torch.stack(tgt_embs, 2)  # [bs, l, n_clusters, emb]
        emb = torch.matmul(cluster_weights.unsqueeze(2), tgt_embs)
        emb = emb.squeeze(2)
        return self.ln(emb) * self.scale


class AdaptiveEmbedding(nn.Module):
    def __init__(self, vocab_size:int, base_embedding_dim:int, projection_dim:int,
                 cutoffs=None, div_val=2):
        super(AdaptiveEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.n_cluster = len(cutoffs) + 1 if cutoffs is not None else 4
        self.projection_dim = projection_dim
        self.scale = projection_dim**0.5
        cutoffs = cutoffs if cutoffs else self.compute_cutoffs()
        self.cutoffs = [0] + cutoffs + [vocab_size]
        self.embedding_dims = [base_embedding_dim // (div_val**i) for i in range(self.n_cluster)]
        # print(self.embedding_dims)
        self.embeddings = nn.ModuleList([nn.Embedding(self.cutoffs[i+1]-self.cutoffs[i],
                                                      self.embedding_dims[i])
                                         if i != self.n_cluster - 1
                                         else nn.Embedding(self.cutoffs[i+1]-self.cutoffs[i] + 1,
                                                           self.embedding_dims[i],self.cutoffs[i+1]-self.cutoffs[i])# for UNK
                                         for i in range(self.n_cluster)])
        self.proj = nn.ModuleList([nn.Linear(i,projection_dim) for i in self.embedding_dims])

    def compute_cutoffs(self):
        target_ratio = [0.01, 0.05, 0.2]
        return [int(i*self.vocab_size) for i in target_ratio]

    def forward(self,x):
        flat_x = x.contiguous().view(-1)
        total_embedding = torch.zeros(flat_x.size(0),self.projection_dim,
                                      device=x.device, dtype=torch.half)
        for i in range(self.n_cluster):
            l,r = self.cutoffs[i], self.cutoffs[i+1]
            if i == self.n_cluster - 1:
                r +=1
            mask = (flat_x >=l) & (flat_x<r)
            indices = mask.nonzero().squeeze()
            if indices.numel() == 0:
                continue
            x_i = flat_x[indices] - l
            target_embedding = self.embeddings[i](x_i)
            projected_embedding = self.proj[i](target_embedding)
            total_embedding[indices] = projected_embedding
        total_embedding = total_embedding.view(*x.size(), self.projection_dim)
        total_embedding.mul_(self.scale)
        return total_embedding


class PositionEmbedding(nn.Module):
    def __init__(self, embedding_dim:int):
        super(PositionEmbedding, self).__init__()

        self.embedding_dim = embedding_dim

        inv_freq = 1 / (10000 ** (torch.arange(0.0, embedding_dim, 2.0) / embedding_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        # pos_seq.size() = (query_lengths)
        if len(pos_seq.size()) ==1:
            sinusoid = torch.ger(pos_seq, self.inv_freq)
        elif len(pos_seq.size()) ==2:
            sinusoid = torch.einsum('ab,c->abc',pos_seq,self.inv_freq)
        pos_emb = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)
        return pos_emb


class HybridEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, padding_index: int, dropout_rate:float):
        super(HybridEmbedding, self).__init__()
        self.unique_embedding = AdaptiveEmbedding(vocab_size, embedding_dim, embedding_dim)
        self.shared_embedding = StructuredEmbedding(vocab_size, embedding_dim)
        self.ratio = nn.Embedding(vocab_size,1)
        self.unique_ln = nn.LayerNorm(embedding_dim)
        self.shared_ln = nn.LayerNorm(embedding_dim)
        self.scale = nn.Parameter(torch.Tensor([0.02]))

    def forward(self, x):
        u = self.unique_ln(self.unique_embedding(x))
        s = self.shared_ln(self.shared_embedding(x))
        r = torch.sigmoid(self.ratio(x))
        res = r * u + (1-r * s)
        return res * self.scale


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, padding_index: int,
                 max_seqlen: int, dropout_rate: float = 0.1, use_pos_emb: bool = True):
        super(TransformerEmbedding, self).__init__()

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.use_pos_emb = use_pos_emb
        self.seq_len = max_seqlen

        # self.word_embedding = HybridEmbedding(vocab_size, embedding_dim, padding_index, dropout_rate)
        # self.word_embedding = StructuredEmbedding(vocab_size, embedding_dim)
        # self.word_embedding = AdaptiveEmbedding(vocab_size, embedding_dim, embedding_dim)
        # self.word_embedding = OneEmbed(vocab_size, embedding_dim, padding_index,
        #                                one_emb_type='real', dropout=dropout_rate)
        # self.word_embedding = HashEmbedding(vocab_size, embedding_dim, padding_index)
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_index)
        # self.pos_ln = nn.LayerNorm(embedding_dim)
        if self.use_pos_emb:
            self.posisition_embedding = nn.Embedding(max_seqlen, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mem):
        bs, qs = x.size()
        ms = mem[0].size(1) if mem is not None else 0
        ks = qs + ms
        emb = self.word_embedding(x)
        if self.use_pos_emb:
            emb *= math.sqrt(self.embedding_dim)
            pos_indicator = torch.arange(ms, ks, 1).clamp_max_(self.seq_len).to(emb.device)
            pos_ebd = self.posisition_embedding(pos_indicator)
            emb = pos_ebd + emb
        emb = self.dropout(emb)
        return emb


class OneEmbed(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx,
                 one_emb_type='binary', dropout=0.1, std=0.0675, codenum=64, codebooknum=8,
                 layernum=1, binary_dropout=0.5):
        super(OneEmbed, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.one_emb_type = one_emb_type
        self.layernum = layernum
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(torch.Tensor(1, embedding_dim))  #embedding for all tokens
        nn.init.normal_(self.weight, std=std)
        self.linear = nn.Sequential(nn.Linear(embedding_dim, embedding_dim * 8), nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(embedding_dim * 8, embedding_dim))
        if self.one_emb_type == 'binary':
            prob = torch.Tensor(codenum, embedding_dim)
            nn.init.constant_(prob, (1 - binary_dropout ** (1.0 / codebooknum)))
            self.masklist = [torch.bernoulli(prob) for _ in range(codebooknum)]
        else:
            mean_m = torch.zeros(codenum, embedding_dim)
            std_m = torch.Tensor(codenum, embedding_dim)
            nn.init.constant_(std_m, std * (codebooknum ** -0.5))
            self.masklist = nn.ParameterList(
                [nn.Parameter(torch.normal(mean_m, std_m), requires_grad=False) for _ in range(codebooknum)])
        self.hash2mask = nn.Parameter(torch.randint(0, codenum, (num_embeddings, codebooknum), dtype=torch.long),
                                      requires_grad=False)
        self.mask = None  # mask for each token

    def construct_mask2each_token(self):
        mask = []
        for i in range(self.hash2mask.size(1)):
            token_hash = self.hash2mask[:, i]
            mask.append(nn.functional.embedding(token_hash, self.masklist[i]))
        mask = sum(mask)
        if self.one_emb_type == 'binary':
            mask.clamp_(0, 1)
        return mask

    def construct_matrix_for_output_layer(self):
        vocab_vec = self.mask.new(range(self.num_embeddings)).long()
        matrix = self.forward(vocab_vec, dropout=0)
        return matrix

    def forward(self, x):
        if self.mask is None:
            self.mask = self.construct_mask2each_token()
        if x.is_cuda and not self.mask.is_cuda:
            self.mask = self.mask.cuda().to(torch.half)
        each_token_mask = nn.functional.embedding(x, self.mask, padding_idx=self.padding_idx)
        embed = each_token_mask * self.weight.expand_as(each_token_mask)
        embed = self.linear(embed)
        return embed


class HashEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, pool_size=1000, num_hash=2):
        super(HashEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.pool = nn.Embedding(pool_size, embedding_dim)
        self.import_params = nn.Parameter(torch.Tensor(num_embeddings, num_hash))
        nn.init.constant_(self.import_params, 0.5)
        self.hash_values = nn.Parameter(torch.randint(0, pool_size, (num_embeddings, num_hash), dtype=torch.long),
                                        requires_grad=False)

    def forward(self, input):
        if input.is_cuda and not self.hash_values.is_cuda:
            self.hash_values = self.hash_values.cuda()
        hash_values = nn.functional.embedding(input, self.hash_values, padding_idx=self.padding_idx)
        import_params = nn.functional.embedding(input, self.import_params, padding_idx=self.padding_idx)  # [bs, l, n_h]
        embed = self.pool(hash_values)  # [bs, l, num_hash, embedding_dim]
        embed = (import_params.unsqueeze(-1) * embed).sum(2)
        return embed
