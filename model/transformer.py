import torch
import torch.nn as nn
from model.embeddings import *
from model.softmax import *
from util.initializer import *
from model.layers import *
from model.ops import reindex_embedding
from model.attention import *
from abc import ABC, abstractmethod


class BaseBlock(nn.Module, ABC):
    def __init__(self, hidden_dim: int, projection_dim: int, n_heads: int, head_dim: int,
                 dropout_rate: float, dropatt_rate: float, pre_lnorm: bool = False, rel_att=True):
        super(BaseBlock, self).__init__()
        self.self_att = RelMultiheadAtt(hidden_dim, n_heads, head_dim, dropout_rate, dropatt_rate, pre_lnorm) \
            if rel_att else MultiheadAtt(hidden_dim, n_heads, head_dim, dropout_rate, dropatt_rate, pre_lnorm)
        self.feedforward = ResidualFF(hidden_dim, projection_dim, dropout_rate, pre_lnorm)

    @abstractmethod
    def forward(self, inp, *args):
        pass


class EncoderBlock(BaseBlock):
    def __init__(self, hidden_dim: int, projection_dim: int, n_heads: int, head_dim: int,
                 dropout_rate: float, dropatt_rate: float, pre_lnorm: bool = False, rel_att=True):
        super(EncoderBlock, self).__init__(hidden_dim, projection_dim, n_heads, head_dim,
                                           dropout_rate, dropatt_rate, pre_lnorm, rel_att)

    def forward(self, inp, *args):
        x, mem, mask = inp
        out, new_mem, att_prob = self.self_att(x, x, mem, mask, *args)
        out = self.feedforward(out)
        return out, new_mem, att_prob


class DecoderBlock(BaseBlock):
    def __init__(self, hidden_dim: int, projection_dim: int, n_heads: int, head_dim: int,
                 dropout_rate: float, dropatt_rate: float, pre_lnorm: bool = False, rel_att=True):
        super(DecoderBlock, self).__init__(hidden_dim, projection_dim, n_heads, head_dim,
                                           dropout_rate, dropatt_rate, pre_lnorm, rel_att)
        self.multihead_att = RelMultiheadAtt(hidden_dim, n_heads, head_dim, dropout_rate, dropatt_rate, pre_lnorm) \
            if rel_att else MultiheadAtt(hidden_dim, n_heads, head_dim, dropout_rate, dropatt_rate, pre_lnorm)

    def forward(self, inp, *args):
        src, tgt, tgt_mem, tgt_mask, tgt_to_src_mask = inp
        out, new_mem, self_att_prob = self.self_att(tgt, tgt, tgt_mem, tgt_mask, *args)
        out, _, inter_att_prob = self.multihead_att(out, src, None, tgt_to_src_mask, *args)  # if src is None, this step is skipped
        out = self.feedforward(out)
        return out, new_mem, self_att_prob, inter_att_prob


class BaseNetwork(nn.Module):
    def __init__(self, hidden_dim: int, projection_dim: int,
                 n_heads: int, head_dim: int, n_layers: int,
                 dropout_rate: float, dropatt_rate: float,
                 pre_lnorm: bool = False, same_lengths: bool = False, rel_att: bool = False,
                 block_type: nn.Module = EncoderBlock, is_bidirectional: bool = False, **kwargs):
        super(BaseNetwork, self).__init__()
        self.vocab_size = kwargs.pop('vocab_size', None)
        self.seq_len = kwargs.pop('seq_len', None)
        self.padding_index = kwargs.pop('padding_index', None)
        self.embedding = kwargs.pop('embedding', None)  # is passed if encoder and decoder share embedding
        self.n_layers = n_layers
        self.same_lengths = same_lengths
        self.rel_att = rel_att
        self.is_bidirectional = is_bidirectional
        self.hidden_dim = hidden_dim
        if self.vocab_size:
            self.use_pos_emb = False if rel_att else True
            if not self.embedding:
                assert self.seq_len
                self.embedding = TransformerEmbedding(self.vocab_size, hidden_dim, self.padding_index,
                                                      self.seq_len, dropout_rate, self.use_pos_emb)
            else:
                assert isinstance(self.embedding, TransformerEmbedding)

        if rel_att:
            self.rw_bias = nn.Parameter(torch.Tensor(n_heads, head_dim))
            self.rr_bias = nn.Parameter(torch.Tensor(n_heads, head_dim))

        # if not self.embedding_equal_hidden:
        #     self.embedding_proj = nn.Linear(word_embedding_dim,hidden_dim,bias=False)
        self.main_nets = nn.ModuleList([block_type(hidden_dim, projection_dim, n_heads, head_dim,
                                                   dropout_rate, dropatt_rate, pre_lnorm, rel_att)
                                        for _ in range(n_layers)])

    def get_mask(self, mem, inp_lens):
        inp_masks = mask_lengths(inp_lens, reverse=True).byte()
        bs, qs = inp_masks.size()
        ms = mem[0].size(1) if mem is not None else 0
        ks = qs + ms
        ones = torch.ones((qs, ks)).byte().to(inp_masks.device)
        if not self.is_bidirectional:
            dec_mask = ones.triu(1 + ms)
        else:
            dec_mask = torch.zeros_like(ones)
        if self.same_lengths:
            dec_mask = dec_mask + ones.tril(-qs)
        if ms:
            inp_masks = torch.cat([torch.zeros(bs, ms, dtype=inp_masks.dtype, device=inp_masks.device), inp_masks], 1)
        mask = (inp_masks.unsqueeze(1) + dec_mask.unsqueeze(0)) > 0
        return mask


class EncoderNetwork(BaseNetwork):
    def __init__(self, hidden_dim: int, projection_dim: int,
                 n_heads: int, head_dim: int, n_layers: int,
                 dropout_rate: float, dropatt_rate: float,
                 pre_lnorm: bool = False, same_lengths: bool = False, rel_att: bool = False, **kwargs):
        super(EncoderNetwork, self).__init__(hidden_dim, projection_dim, n_heads, head_dim, n_layers,
                                             dropout_rate, dropatt_rate,
                                             pre_lnorm, same_lengths, rel_att,
                                             EncoderBlock, True, **kwargs)

    def forward(self, x, mem, mask):
        """
        :param x: input, input.size() = [batch_size, seq_len]
        :param mem: list of memories [mem1,mem2, ...memn], n equal to the number of layers
          memory[0].size() = [batch_size, memory_len, hidden_size]
        :param mask: input mask, size = [batch_size, seq_len, seq_len]
        :return:
        """
        emb = self.embedding(x, mem)
        out = emb
        new_mems = []
        enc_self_atts = []
        for i in range(self.n_layers):
            block = self.main_nets[i]
            mem_i = mem[i] if mem is not None else None
            out, new_mem, self_att = block((out, mem_i, mask))
            new_mems.append(new_mem)
            enc_self_atts.append(self_att)
        return out, new_mems, enc_self_atts


class DecoderNetwork(BaseNetwork):
    def __init__(self, hidden_dim: int, projection_dim: int,
                 n_heads: int, head_dim: int, n_layers: int,
                 dropout_rate: float, dropatt_rate: float,
                 pre_lnorm: bool = False, same_lengths: bool = False, rel_att: bool = False, **kwargs):
        super(DecoderNetwork, self).__init__(hidden_dim, projection_dim, n_heads, head_dim, n_layers,
                                             dropout_rate, dropatt_rate,
                                             pre_lnorm, same_lengths, rel_att,
                                             DecoderBlock, False, **kwargs)

    @staticmethod
    def tgt_to_src_mask(src_len, tgt_len):
        src_masks = mask_lengths(src_len, reverse=True).byte()  # bs, sl
        tgt_masks = mask_lengths(tgt_len, reverse=True).byte()
        bs, tl = tgt_masks.size()
        sl = src_masks.size(1)
        zeros = torch.zeros(size=(tl, sl)).to(src_masks.device)
        res = zeros[None] + src_masks[:,None]
        return res.byte()

    def forward(self, src, tgt, mem, tgt_mask, tgt_to_src_mask):
        """
        :param src: final hidden states from source texts [Bs, len, hidden_size]
        :param tgt: input indice of target texts [Bs, len]
        :param mem: cached states for decoder
        :param tgt_mask: input masks of source sentence [Bs, tgt_len, tgt_len]
        :param tgt_to_src_mask: input masks of target sentence [Bs, tgt_len, src_len]
        :return:
        """
        emb = self.embedding(tgt, mem)
        out = emb
        new_mems = []
        self_att_probs = []
        inter_att_probs = []
        for i in range(self.n_layers):
            block = self.main_nets[i]
            mem_i = mem[i] if mem is not None else None
            main_inp = src, out, mem_i, tgt_mask, tgt_to_src_mask
            out, new_mem, self_att_prob, inter_att_prob = block(main_inp)
            new_mems.append(new_mem)
            self_att_probs.append(self_att_prob)
            inter_att_probs.append(inter_att_prob)
        return out, new_mems, self_att_probs, inter_att_probs


class EncoderDecoderModel(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, hidden_dim: int, projection_dim: int, n_heads: int, head_dim: int,
                 enc_num_layers: int, dec_num_layers: int, dropout_rate: float, dropatt_rate: float, padding_index: int,
                 pre_lnorm: bool = False, same_lengths: bool = False, rel_att: bool = False, shared_embedding=False,
                 tie_embedding=False, **kwargs):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = EncoderNetwork(hidden_dim, projection_dim, n_heads, head_dim, enc_num_layers, dropout_rate,
                                      dropatt_rate, pre_lnorm, same_lengths, rel_att,
                                      vocab_size=vocab_size, seq_len=seq_len, padding_index=padding_index)
        self.tie_embedding = tie_embedding
        if shared_embedding:
            kwargs_dict = {'embedding': self.encoder.embedding}
        else:
            kwargs_dict = {'vocab_size': vocab_size, 'seq_len': seq_len, 'padding_index': padding_index}

        self.decoder = DecoderNetwork(hidden_dim, projection_dim, n_heads, head_dim, dec_num_layers, dropout_rate,
                                      dropatt_rate, pre_lnorm, same_lengths, rel_att, **kwargs_dict)
        if self.tie_embedding:
            embedding_weight = self.encoder.embedding.word_embedding.weight
            self.final = lambda x: torch.matmul(x, embedding_weight.T)

        else:
            self.final = nn.Linear(hidden_dim, vocab_size, bias=False)  # To-Do : implement tie embedding

    def encode_src(self, inp):
        src, src_len = inp['src'], inp['src_len']
        src_mask = self.encoder.get_mask(None, src_len)
        enc_out, _, enc_self_atts = self.encoder(src, None, src_mask)
        return {'enc_out': enc_out, 'enc_self_att': enc_self_atts}

    def forward(self, inp):
        src, tgt, src_len, tgt_len = inp['src'], inp['tgt'], inp['src_len'], inp['tgt_len']
        enc_out = inp['enc_out'] if 'enc_out' in inp else None  # if src is already encoded. used in decoding phase
        tgt_mem = inp['tgt_mem'] if 'tgt_mem' in inp else None
        out = {}
        if enc_out is None:
            from_enc = self.encode_src(inp)
            out['enc_out'] = from_enc['enc_out']
            out['enc_self_att'] = from_enc['enc_self_att']
            enc_out = from_enc['enc_out']
        tgt_mask = self.decoder.get_mask(tgt_mem, tgt_len)
        tgt_to_src_mask = self.decoder.tgt_to_src_mask(src_len, tgt_len)
        dec_out, new_tgt_mem, dec_self_att, inter_att = self.decoder(enc_out, tgt, tgt_mem, tgt_mask, tgt_to_src_mask)
        logits = self.final(dec_out)
        out['logits'] = logits
        out['tgt_mem'] = new_tgt_mem
        out['dec_self_att'] = dec_self_att
        out['inter_att'] = inter_att
        return out


class LMModel(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, hidden_dim: int, projection_dim: int,
                 n_heads: int, head_dim: int, n_layers: int,
                 dropout_rate: float, dropatt_rate: float, padding_index: int,
                 pre_lnorm: bool = False, same_lengths: bool = False, rel_att: bool = True, ):
        super(LMModel, self).__init__()
        self.transformer = DecoderNetwork(hidden_dim, projection_dim, n_heads, head_dim, n_layers,
                                          dropout_rate, dropatt_rate, pre_lnorm, same_lengths, rel_att,
                                          vocab_size=vocab_size, seq_len=seq_len, padding_index=padding_index)
        self.final = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, inp):
        x, inp_lens, mem = inp
        bs, qs = x.size()
        out, mem = self.transformer(x, mem)
        out = out[:, :-1]
        out = out.contiguous().view(bs * (qs - 1), -1)
        final = self.final(out)
        return final, mem


class CrossLingualModel(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, hidden_dim: int, projection_dim: int, n_heads: int, head_dim: int,
                 enc_num_layers: int, dec_num_layers: int, dropout_rate: float, dropatt_rate: float, padding_index: int,
                 pre_lnorm: bool = False, same_lengths: bool = False, rel_att: bool = False, shared_embedding=False,
                 tie_embedding=False, num_decoders=2, **kwargs):
        super(CrossLingualModel, self).__init__()
        self.encoder = EncoderNetwork(hidden_dim, projection_dim, n_heads, head_dim, enc_num_layers, dropout_rate,
                                      dropatt_rate, pre_lnorm, same_lengths, rel_att,
                                      vocab_size=vocab_size, seq_len=seq_len, padding_index=padding_index)
        self.tie_embedding = tie_embedding
        self.num_decoders = num_decoders
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        if shared_embedding:
            kwargs_dict = {'embedding': self.encoder.embedding}
        else:
            kwargs_dict = {'vocab_size': vocab_size, 'seq_len': seq_len, 'padding_index': padding_index}

        self.decoder = nn.ModuleList([DecoderNetwork(hidden_dim, projection_dim, n_heads, head_dim,
                                                     dec_num_layers, dropout_rate, dropatt_rate, pre_lnorm,
                                                     same_lengths, rel_att, **kwargs_dict) for i in range(num_decoders)])
        if self.tie_embedding:
            raise NotImplementedError
        else:
            self.final = nn.ModuleList([nn.Linear(hidden_dim, vocab_size, bias=False) for i in range(num_decoders)])

    def encode_src(self, inp):
        src, src_len = inp['src'], inp['src_len']
        src_mask = self.encoder.get_mask(None, src_len)
        enc_out, _, enc_self_atts = self.encoder(src, None, src_mask)
        return {'enc_out': enc_out, 'enc_self_att': enc_self_atts}

    def forward(self, inp):
        src, tgt, src_len, tgt_len, tgt_language = inp['src'], inp['tgt'], inp['src_len'],\
                                                   inp['tgt_len'], inp['tgt_language']
        enc_out = inp['enc_out'] if 'enc_out' in inp else None  # if src is already encoded. used in decoding phase
        tgt_mem = inp['tgt_mem'] if 'tgt_mem' in inp else None
        out = {}
        if enc_out is None:
            from_enc = self.encode_src(inp)
            out['enc_out'] = from_enc['enc_out']
            out['enc_self_att'] = from_enc['enc_self_att']
            enc_out = from_enc['enc_out']
        tgt_mask = self.decoder[0].get_mask(tgt_mem, tgt_len)
        tgt_to_src_mask = self.decoder[0].tgt_to_src_mask(src_len, tgt_len)
        bs, l = tgt.size()
        z_logits = torch.zeros(size=(bs, l, self.vocab_size), dtype=enc_out.dtype, device=enc_out.device)
        z_tgt_mem = [torch.zeros(size=(bs, l, self.hidden_dim*2), dtype=enc_out.dtype, device=enc_out.device)]

        for i in range(self.num_decoders):
            ind = (tgt_language == i).nonzero(as_tuple=False).squeeze(1)
            if ind.numel() == 0:
                continue
            i_tgt_mask = tgt_mask[ind]
            i_tgt_to_src_mask = tgt_to_src_mask[ind]
            i_enc_out = enc_out[ind]
            i_tgt = tgt[ind]
            i_tgt_mem = [k[ind] for k in tgt_mem] if tgt_mem is not None else None
            i_dec_out, i_new_tgt_mem, dec_self_att, inter_att = self.decoder[i](i_enc_out, i_tgt,i_tgt_mem,
                                                                                i_tgt_mask, i_tgt_to_src_mask)
            i_logits = self.final[i](i_dec_out)
            z_logits[ind] = i_logits
            for j in range(len(z_tgt_mem)):
                z_tgt_mem[j][ind] = i_new_tgt_mem[j]
        out['logits'] = z_logits
        out['tgt_mem'] = z_tgt_mem
        out['dec_self_att'] = dec_self_att
        out['inter_att'] = inter_att
        return out


class ComplexityAwareModel(EncoderDecoderModel):
    def __init__(self, vocab_size: int, seq_len: int, hidden_dim: int, projection_dim: int, n_heads: int, head_dim: int,
                 enc_num_layers: int, dec_num_layers: int, dropout_rate: float, dropatt_rate: float, padding_index: int,
                 n_cluster:int, pre_lnorm: bool = False, same_lengths: bool = False, rel_att: bool = False,
                 shared_embedding=False, tie_embedding=False, **kwargs):
        super(ComplexityAwareModel, self).__init__(vocab_size, seq_len, hidden_dim, projection_dim, n_heads,
                                                   head_dim, enc_num_layers, dec_num_layers, dropout_rate,
                                                   dropatt_rate, padding_index, pre_lnorm, same_lengths,
                                                   rel_att, shared_embedding, tie_embedding, ** kwargs)
        delattr(self, 'final')
        self.final = ComplexityControllingSoftmax(vocab_size, hidden_dim)
