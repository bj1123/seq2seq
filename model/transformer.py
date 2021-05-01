import torch
import torch.nn as nn
from model.embeddings import *
from model.softmax import *
from util.initializer import *
from model.layers import *
from model.ops import reindex_embedding
from model.attention import *
from abc import ABC, abstractmethod
from model.modules import TextConv


class BaseBlock(nn.Module, ABC):
    def __init__(self, hidden_dim: int, projection_dim: int, n_heads: int, head_dim: int,
                 dropout_rate: float, dropatt_rate: float, pre_lnorm: bool = False,
                 att_module=MultiheadAtt, **kwargs):
        super(BaseBlock, self).__init__()
        self.self_att = att_module(hidden_dim, n_heads, head_dim, dropout_rate, dropatt_rate, pre_lnorm, **kwargs)
        self.feedforward = ResidualFF(hidden_dim, projection_dim, dropout_rate, pre_lnorm)

    @abstractmethod
    def forward(self, inp, *args):
        pass


class EncoderBlock(BaseBlock):
    def __init__(self, hidden_dim: int, projection_dim: int, n_heads: int, head_dim: int,
                 dropout_rate: float, dropatt_rate: float, pre_lnorm: bool = False, att_module=MultiheadAtt,
                 **kwargs):
        super(EncoderBlock, self).__init__(hidden_dim, projection_dim, n_heads, head_dim,
                                           dropout_rate, dropatt_rate, pre_lnorm, att_module, **kwargs)

    def forward(self, inp, *args):
        x, mem, mask = inp
        out, new_mem, att_prob = self.self_att(x, x, mem, mask, *args)
        out = self.feedforward(out)
        return out, new_mem, att_prob


class DecoderBlock(BaseBlock):
    def __init__(self, hidden_dim: int, projection_dim: int, n_heads: int, head_dim: int,
                 dropout_rate: float, dropatt_rate: float, pre_lnorm: bool = False,
                 att_module=MultiheadAtt, **kwargs):
        super(DecoderBlock, self).__init__(hidden_dim, projection_dim, n_heads, head_dim,
                                           dropout_rate, dropatt_rate, pre_lnorm, att_module, **kwargs)
        self.multihead_att = att_module(hidden_dim, n_heads, head_dim, dropout_rate, dropatt_rate, pre_lnorm, **kwargs)

    def forward(self, inp, *args):
        src, tgt, tgt_mem, tgt_mask, tgt_to_src_mask = inp
        out, new_mem, self_att_prob = self.self_att(tgt, tgt, tgt_mem, tgt_mask, *args)
        out, _, inter_att_prob = self.multihead_att(out, src, None, tgt_to_src_mask,
                                                    *args)  # if src is None, this step is skipped
        out = self.feedforward(out)
        return out, new_mem, self_att_prob, inter_att_prob


class AttentionSpecificEncoderBlock(BaseBlock):
    def __init__(self, hidden_dim: int, projection_dim: int, n_heads: int, head_dim: int,
                 dropout_rate: float, dropatt_rate: float, pre_lnorm: bool = False,
                 att_module=LanguageWiseAttention, **kwargs):
        super(AttentionSpecificEncoderBlock, self).__init__(hidden_dim, projection_dim, n_heads, head_dim,
                                                            dropout_rate, dropatt_rate, pre_lnorm, att_module,
                                                            **self.convert_kwargs(kwargs))

    @staticmethod
    def convert_kwargs(kwargs):
        num_src_lang = kwargs.get('num_src_lang')
        kwargs['num_tgt_lang'] = num_src_lang
        return kwargs

    def forward(self, inp, *args):
        x, mem, mask = inp
        src_lang = args[0]
        out, new_mem, att_prob = self.self_att(x, x, mem, mask, src_lang, src_lang)
        out = self.feedforward(out)
        return out, new_mem, att_prob


class AttentionSpecificDecoderBlock(BaseBlock):
    def __init__(self, hidden_dim: int, projection_dim: int, n_heads: int, head_dim: int,
                 dropout_rate: float, dropatt_rate: float, pre_lnorm: bool = False,
                 att_module=LanguageWiseAttention, **kwargs):
        super(AttentionSpecificDecoderBlock, self).__init__(hidden_dim, projection_dim, n_heads, head_dim,
                                                            dropout_rate, dropatt_rate, pre_lnorm, att_module,
                                                            **self.selfatt_kwargs(kwargs))
        self.multihead_att = att_module(hidden_dim, n_heads, head_dim, dropout_rate, dropatt_rate, pre_lnorm,
                                        **self.multiatt_kwargs(kwargs))

    @staticmethod
    def selfatt_kwargs(kwargs):
        num_tgt_lang = kwargs.get('num_tgt_lang')
        new_dic = kwargs.copy()
        new_dic['num_src_lang'] = num_tgt_lang
        return new_dic

    @staticmethod
    def multiatt_kwargs(kwargs):
        num_tgt_lang = kwargs.get('num_tgt_lang')
        num_src_lang = kwargs.get('num_src_lang')
        new_dic = kwargs.copy()
        new_dic['num_src_lang'] = num_tgt_lang
        new_dic['num_tgt_lang'] = num_src_lang
        return new_dic

    def forward(self, inp, *args):
        src, tgt, tgt_mem, tgt_mask, tgt_to_src_mask = inp
        src_lang, tgt_lang = args[0], args[1]
        out, new_mem, self_att_prob = self.self_att(tgt, tgt, tgt_mem, tgt_mask, tgt_lang, tgt_lang)
        out, _, inter_att_prob = self.multihead_att(out, src, None, tgt_to_src_mask,
                                                    tgt_lang, src_lang)  # if src is None, this step is skipped
        out = self.feedforward(out)
        return out, new_mem, self_att_prob, inter_att_prob


class SentenceAwareDecoderBlock(DecoderBlock):
    def __init__(self, hidden_dim: int, projection_dim: int, n_heads: int, head_dim: int,
                 dropout_rate: float, dropatt_rate: float, pre_lnorm: bool = False, att_module=MultiheadAtt,
                 **kwargs):
        super(SentenceAwareDecoderBlock, self).__init__(hidden_dim, projection_dim, n_heads, head_dim,
                                                        dropout_rate, dropatt_rate, pre_lnorm, att_module)
        self.sentence_att = SentenceAwareAtt(hidden_dim, n_heads, head_dim, dropout_rate,
                                             dropatt_rate, pre_lnorm, **kwargs)

    def forward(self, inp, *args):
        src, tgt, tgt_mem, tgt_mask, tgt_to_src_mask, tgt_emb = inp
        if tgt_mem is None:
            tgt_mem_self, tgt_mem_sent = None, None
        else:
            tgt_mem_self, tgt_mem_sent = tgt_mem.chunk(2, dim=-1)
        out, new_tgt_mem_self, self_att_prob = self.self_att(tgt, tgt, tgt_mem_self, tgt_mask, *args)
        tgt_emb, new_tgt_mem_sent, _ = self.sentence_att(tgt_emb, tgt, tgt_mem_sent, tgt_mask, *args)
        out, _, inter_att_prob = self.multihead_att(out - tgt_emb, src, None, tgt_to_src_mask,
                                                    *args)  # if src is None, this step is skipped
        out = self.feedforward(out)
        return out, torch.cat([new_tgt_mem_self, new_tgt_mem_sent], dim=-1), self_att_prob, inter_att_prob


class BaseNetwork(nn.Module):
    def __init__(self, hidden_dim: int, projection_dim: int,
                 n_heads: int, head_dim: int, n_layers: int,
                 dropout_rate: float, dropatt_rate: float,
                 pre_lnorm: bool = False, same_lengths: bool = False, pos_enc: str = 'absolute',
                 block_type: nn.Module = EncoderBlock, is_bidirectional: bool = False, use_embedding=True, **kwargs):
        super(BaseNetwork, self).__init__()
        self.kwargs = kwargs
        self.vocab_size = kwargs.get('vocab_size', None)
        self.seq_len = kwargs.get('seq_len', None)
        self.padding_index = kwargs.get('padding_index', None)
        self.embedding = kwargs.get('embedding', None)  # is passed if encoder and decoder share embedding
        self.att_module = kwargs.pop('att_module', MultiheadAtt)
        self.n_layers = n_layers
        self.same_lengths = same_lengths
        self.is_bidirectional = is_bidirectional
        self.hidden_dim = hidden_dim
        self.use_embedding = use_embedding
        if use_embedding:
            if self.vocab_size:
                self.use_pos_emb = 'absolute' in pos_enc
                if not self.embedding:
                    assert self.seq_len
                    self.embedding = TransformerEmbedding(self.vocab_size, hidden_dim, self.padding_index,
                                                          self.seq_len, dropout_rate, self.use_pos_emb)
                else:
                    assert isinstance(self.embedding, TransformerEmbedding)
        # if not self.embedding_equal_hidden:
        #     self.embedding_proj = nn.Linear(word_embedding_dim,hidden_dim,bias=False)

        pos_encs = [self.att_module for _ in range(n_layers)]
        if pos_enc == 'relative':
            pos_encs[0] = RelMultiheadAtt
        elif pos_enc == 'graph-relative':
            pos_encs[0] = GraphRelMultiheadAtt
            # pos_encs = [RelMultiheadAtt for _ in range(n_layers)]
        key_args = self.get_relative_encoding_args(pos_enc)
        self.main_nets = nn.ModuleList([block_type(hidden_dim, projection_dim, n_heads, head_dim,
                                                   dropout_rate, dropatt_rate, pre_lnorm, pos_encs[i], **key_args)
                                        for i in range(n_layers)])

    def get_mask(self, mem, inp_lens):
        inp_masks = mask_lengths(inp_lens, reverse=True).byte()
        bs, qs = inp_masks.size()
        ms = mem[0].size(1) if mem is not None else 0
        ks = qs + ms
        ones = torch.ones((qs, ks), dtype=torch.uint8, device=inp_masks.device)
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

    def get_relative_encoding_args(self, pos_enc):
        if pos_enc in ['relative', 'graph-relative']:
            d = {'maxlen': self.seq_len, 'relative_attention_num_buckets': self.seq_len // 4,
                 'is_decoder': not self.is_bidirectional}
        else:
            d = {}
        self.kwargs.update(d)
        return self.kwargs


class EncoderNetwork(BaseNetwork):
    def __init__(self, hidden_dim: int, projection_dim: int, n_heads: int, head_dim: int, n_layers: int,
                 dropout_rate: float, dropatt_rate: float, pre_lnorm: bool = False, same_lengths: bool = False,
                 pos_enc: str = 'absolute', use_embedding=True, encoder_block=EncoderBlock, **kwargs):
        super(EncoderNetwork, self).__init__(hidden_dim, projection_dim, n_heads, head_dim, n_layers,
                                             dropout_rate, dropatt_rate,
                                             pre_lnorm, same_lengths, pos_enc,
                                             encoder_block, True, use_embedding, **kwargs)

    def forward(self, x, mem, mask, *args):
        """
        :param x: input, input.size() = [batch_size, seq_len]
        :param mem: list of memories [mem1,mem2, ...memn], n equal to the number of layers
          memory[0].size() = [batch_size, memory_len, hidden_size]
        :param mask: input mask, size = [batch_size, seq_len, seq_len]
        :return:
        """
        if self.use_embedding:
            emb = self.embedding(x, mem)
            out = emb
        else:
            out = x
        new_mems = []
        enc_self_atts = []
        for i in range(self.n_layers):
            block = self.main_nets[i]
            mem_i = mem[i] if mem is not None else None
            out, new_mem, self_att = block((out, mem_i, mask), *args)
            new_mems.append(new_mem)
            enc_self_atts.append(self_att)
        return out, new_mems, enc_self_atts


class DecoderNetwork(BaseNetwork):
    def __init__(self, hidden_dim: int, projection_dim: int,
                 n_heads: int, head_dim: int, n_layers: int,
                 dropout_rate: float, dropatt_rate: float,
                 pre_lnorm: bool = False, same_lengths: bool = False, pos_enc: str = 'absolute', use_embedding=True,
                 decoder_block=DecoderBlock, **kwargs):
        super(DecoderNetwork, self).__init__(hidden_dim, projection_dim, n_heads, head_dim, n_layers,
                                             dropout_rate, dropatt_rate,
                                             pre_lnorm, same_lengths, pos_enc,
                                             decoder_block, False, use_embedding, **kwargs)

    @staticmethod
    def tgt_to_src_mask(src_len, tgt_len):
        src_masks = mask_lengths(src_len, reverse=True).byte()  # bs, sl
        tgt_masks = mask_lengths(tgt_len, reverse=True).byte()
        bs, tl = tgt_masks.size()
        sl = src_masks.size(1)
        zeros = torch.zeros(size=(tl, sl), device=src_masks.device)
        res = zeros[None] + src_masks[:, None]
        return res.byte()

    def forward(self, src, tgt, mem, tgt_mask, tgt_to_src_mask, *args):
        """
        :param src: final hidden states from source texts [Bs, len, hidden_size]
        :param tgt: input indice of target texts [Bs, len]
        :param mem: cached states for decoder
        :param tgt_mask: input masks of source sentence [Bs, tgt_len, tgt_len]
        :param tgt_to_src_mask: input masks of target sentence [Bs, tgt_len, src_len]
        :return:
        """
        if self.use_embedding:
            emb = self.embedding(tgt, mem)
            out = emb
        else:
            out = tgt
        new_mems = []
        self_att_probs = []
        inter_att_probs = []
        for i in range(self.n_layers):
            block = self.main_nets[i]
            mem_i = mem[i] if mem is not None else None
            main_inp = (src, out, mem_i, tgt_mask, tgt_to_src_mask)
            out, new_mem, self_att_prob, inter_att_prob = block(main_inp, *args)
            new_mems.append(new_mem)
            self_att_probs.append(self_att_prob)
            inter_att_probs.append(inter_att_prob)
        return out, new_mems, self_att_probs, inter_att_probs


class EncoderDecoderBase(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, hidden_dim: int, projection_dim: int, n_heads: int, head_dim: int,
                 enc_num_layers: int, dec_num_layers: int, dropout_rate: float, dropatt_rate: float, padding_index: int,
                 pre_lnorm: bool = False, same_lengths: bool = False, pos_enc: str = 'absolute', shared_embedding=False,
                 tie_embedding=False, **kwargs):
        super(EncoderDecoderBase, self).__init__()
        self.tie_embedding = tie_embedding
        self.shared_embedding = shared_embedding
        self.padding_index = padding_index
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.dec_num_layers = dec_num_layers
        self.enc_num_layers = enc_num_layers
        self.dropout_rate = dropout_rate
        self.dropatt_rate = dropatt_rate
        self.pre_lnorm = pre_lnorm
        self.same_lengths = same_lengths
        self.pos_enc = pos_enc

    def keyword_dict(self):
        kwargs_dict = {'vocab_size': self.vocab_size, 'seq_len': self.seq_len, 'padding_index': self.padding_index}
        if self.shared_embedding and hasattr(self, 'encoder'):
            kwargs_dict['embedding'] = self.encoder.embedding
        elif self.shared_embedding and hasattr(self, 'shared_encoders'):
            kwargs_dict['embedding'] = self.shared_encoders.embedding
        return kwargs_dict

    def build_decoder_network(self, block_type=DecoderBlock):
        return DecoderNetwork(self.hidden_dim, self.projection_dim, self.n_heads, self.head_dim, self.dec_num_layers,
                              self.dropout_rate, self.dropatt_rate, self.pre_lnorm, self.same_lengths,
                              self.pos_enc, decoder_block=block_type, **self.keyword_dict())


class EncoderDecoderModel(EncoderDecoderBase):
    def __init__(self, vocab_size: int, seq_len: int, hidden_dim: int, projection_dim: int, n_heads: int, head_dim: int,
                 enc_num_layers: int, dec_num_layers: int, dropout_rate: float, dropatt_rate: float, padding_index: int,
                 pre_lnorm: bool = False, same_lengths: bool = False, pos_enc: str = 'absolute', shared_embedding=False,
                 tie_embedding=False, **kwargs):
        super(EncoderDecoderModel, self).__init__(vocab_size, seq_len, hidden_dim, projection_dim, n_heads, head_dim,
                                                  enc_num_layers, dec_num_layers, dropout_rate, dropatt_rate,
                                                  padding_index,
                                                  pre_lnorm, same_lengths, pos_enc, shared_embedding, tie_embedding,
                                                  **kwargs)
        self.encoder = EncoderNetwork(hidden_dim, projection_dim, n_heads, head_dim, enc_num_layers, dropout_rate,
                                      dropatt_rate, pre_lnorm, same_lengths, pos_enc,
                                      **self.keyword_dict())
        self.decoder = self.build_decoder_network()
        if self.tie_embedding:
            embedding_weight = self.decoder.embedding.word_embedding.weight
            self.final = lambda x: torch.matmul(x, embedding_weight.T)

        else:
            self.final = SemiAdaptiveSoftmax(vocab_size, hidden_dim)
            # self.final = nn.Linear(hidden_dim, vocab_size, bias=False)

    def encode_src(self, inp):
        src, src_len = inp['src'], inp['src_len']
        src_mask = self.encoder.get_mask(None, src_len)
        enc_out, _, enc_self_atts = self.encoder(src, None, src_mask)
        inp['enc_out'] = enc_out
        inp['enc_self_att'] = enc_self_atts
        return enc_out

    def forward(self, inp):
        src, tgt, src_len, tgt_len = inp['src'], inp['tgt'], inp['src_len'], inp['tgt_len']
        enc_out = inp['enc_out'] if 'enc_out' in inp else None  # if src is already encoded. used in decoding phase
        tgt_mem = inp['tgt_mem'] if 'tgt_mem' in inp else None
        out = {}
        if enc_out is None:
            enc_out = self.encode_src(inp)
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
        self.dec_num_layers = dec_num_layers
        self.num_decoders = num_decoders
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        if shared_embedding:
            kwargs_dict = {'embedding': self.encoder.embedding}
        else:
            kwargs_dict = {'vocab_size': vocab_size, 'seq_len': seq_len, 'padding_index': padding_index}

        self.decoder = nn.ModuleList([DecoderNetwork(hidden_dim, projection_dim, n_heads, head_dim,
                                                     dec_num_layers, dropout_rate, dropatt_rate, pre_lnorm,
                                                     same_lengths, rel_att, **kwargs_dict) for i in
                                      range(num_decoders)])
        if self.tie_embedding:
            raise NotImplementedError
        else:
            self.final = nn.ModuleList([nn.Linear(hidden_dim, vocab_size, bias=False) for i in range(num_decoders)])

    def encode_src(self, inp):
        src, src_len = inp['src'], inp['src_len']
        src_mask = self.encoder.get_mask(None, src_len)
        enc_out, _, enc_self_atts = self.encoder(src, None, src_mask)
        inp['enc_out'] = enc_out
        inp['enc_self_att'] = enc_self_atts
        return enc_out

    def forward(self, inp):
        src, tgt, src_len, tgt_len, tgt_language = inp['src'], inp['tgt'], inp['src_len'], \
                                                   inp['tgt_len'], inp['tgt_language']
        enc_out = inp['enc_out'] if 'enc_out' in inp else None  # if src is already encoded. used in decoding phase
        tgt_mem = inp['tgt_mem'] if 'tgt_mem' in inp else None
        out = {}
        if enc_out is None:
            enc_out = self.encode_src(inp)
        tgt_mask = self.decoder[0].get_mask(tgt_mem, tgt_len)
        tgt_to_src_mask = self.decoder[0].tgt_to_src_mask(src_len, tgt_len)
        bs, l = tgt.size()
        z_logits = torch.zeros(size=(bs, l, self.vocab_size), dtype=enc_out.dtype, device=enc_out.device)
        z_tgt_mem = [torch.zeros(size=(bs, l, self.hidden_dim * 2), dtype=enc_out.dtype, device=enc_out.device)
                     for _ in range(self.dec_num_layers)]
        for i in range(self.num_decoders):
            ind = (tgt_language == i).nonzero(as_tuple=False).squeeze(1)
            if ind.numel() == 0:
                continue
            i_tgt_mask = tgt_mask[ind]
            i_tgt_to_src_mask = tgt_to_src_mask[ind]
            i_enc_out = enc_out[ind]
            i_tgt = tgt[ind]
            i_tgt_mem = [k[ind] for k in tgt_mem] if tgt_mem is not None else None
            i_dec_out, i_new_tgt_mem, dec_self_att, inter_att = self.decoder[i](i_enc_out, i_tgt, i_tgt_mem,
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


class AttentionSpecificMT(EncoderDecoderBase):
    def __init__(self, vocab_size: int, seq_len: int, hidden_dim: int, projection_dim: int, n_heads: int, head_dim: int,
                 enc_num_layers: int, dec_num_layers: int, num_src_lang: int, num_tgt_lang: int,
                 dropout_rate: float, dropatt_rate: float, padding_index: int,
                 pre_lnorm: bool = False, same_lengths: bool = False, pos_enc: str = 'absolute', shared_embedding=False,
                 tie_embedding=False, **kwargs):
        super(AttentionSpecificMT, self).__init__(vocab_size, seq_len, hidden_dim, projection_dim, n_heads,
                                                  head_dim, enc_num_layers, dec_num_layers, dropout_rate,
                                                  dropatt_rate, padding_index, pre_lnorm, same_lengths,
                                                  pos_enc, shared_embedding, tie_embedding, **kwargs)
        self.kwargs = kwargs
        self.num_src_lang = num_src_lang
        self.num_tgt_lang = num_tgt_lang
        self.encoder = EncoderNetwork(hidden_dim, projection_dim, n_heads, head_dim, enc_num_layers,
                                      dropout_rate, dropatt_rate, pre_lnorm, same_lengths, pos_enc,
                                      encoder_block=AttentionSpecificEncoderBlock,
                                      **self.keyword_dict())
        self.decoder = DecoderNetwork(hidden_dim, projection_dim, n_heads, head_dim, dec_num_layers,
                                      dropout_rate, dropatt_rate, pre_lnorm, same_lengths, pos_enc,
                                      decoder_block=AttentionSpecificDecoderBlock,
                                      **self.keyword_dict())
        self.final = nn.Linear(hidden_dim, vocab_size)

    def keyword_dict(self):
        kwargs = super().keyword_dict()
        kwargs['num_src_lang'] = self.num_src_lang
        kwargs['num_tgt_lang'] = self.num_tgt_lang
        kwargs['att_module'] = LanguageWiseAttention
        return kwargs

    def encode_src(self, inp):
        src, src_len, src_lang = inp['src'], inp['src_len'], inp['src_lang']
        src_lang = src_lang[0]  # brute-force implementation
        src_mask = self.encoder.get_mask(None, src_len)
        enc_out, _, enc_self_atts = self.encoder(src, None, src_mask, src_lang, src_lang)
        inp['enc_out'] = enc_out
        inp['enc_self_att'] = enc_self_atts
        return enc_out

    def forward(self, inp):
        src, tgt, src_len, tgt_len, src_lang, tgt_lang = \
            inp['src'], inp['tgt'], inp['src_len'], inp['tgt_len'], inp['src_lang'], inp['tgt_lang']

        src_lang = src_lang[0]  # brute-force implementation
        tgt_lang = tgt_lang[0]  # brute-force implementation
        enc_out = inp['enc_out'] if 'enc_out' in inp else None  # if src is already encoded. used in decoding phase
        tgt_mem = inp['tgt_mem'] if 'tgt_mem' in inp else None
        out = {}
        if enc_out is None:
            enc_out = self.encode_src(inp)
        tgt_mask = self.decoder.get_mask(tgt_mem, tgt_len)
        tgt_to_src_mask = self.decoder.tgt_to_src_mask(src_len, tgt_len)
        dec_out, new_tgt_mem, dec_self_att, inter_att = self.decoder(enc_out, tgt, tgt_mem, tgt_mask, tgt_to_src_mask,
                                                                     src_lang, tgt_lang)

        logits = self.final(dec_out)
        out['logits'] = logits
        out['tgt_mem'] = new_tgt_mem
        out['dec_self_att'] = dec_self_att
        out['inter_att'] = inter_att
        return out


class LanguageSpecificMT(EncoderDecoderBase):
    def __init__(self, vocab_size: int, seq_len: int, hidden_dim: int, projection_dim: int, n_heads: int, head_dim: int,
                 enc_num_layers: int, dec_num_layers: int, num_src_lang: int, num_tgt_lang: int,
                 dropout_rate: float, dropatt_rate: float, padding_index: int,
                 pre_lnorm: bool = False, same_lengths: bool = False, pos_enc: str = 'absolute', shared_embedding=False,
                 tie_embedding=False, shared_enc_layers=None, shared_dec_layers=None, **kwargs):
        super(LanguageSpecificMT, self).__init__(vocab_size, seq_len, hidden_dim, projection_dim, n_heads,
                                                 head_dim, enc_num_layers, dec_num_layers, dropout_rate,
                                                 dropatt_rate, padding_index, pre_lnorm, same_lengths,
                                                 pos_enc, shared_embedding, tie_embedding, **kwargs)
        self.shared_enc_layers = shared_enc_layers if shared_enc_layers else enc_num_layers // 3 * 2
        self.shared_dec_layers = shared_dec_layers if shared_dec_layers else dec_num_layers // 3 * 2
        self.specific_enc_layers = enc_num_layers - self.shared_enc_layers
        self.specific_dec_layers = dec_num_layers - self.shared_dec_layers
        self.num_src_lang = num_src_lang
        self.num_tgt_lang = num_tgt_lang
        self.shared_encoders = EncoderNetwork(hidden_dim, projection_dim, n_heads, head_dim, self.shared_enc_layers,
                                              dropout_rate, dropatt_rate, pre_lnorm, same_lengths, pos_enc,
                                              **self.keyword_dict())
        self.shared_decoders = DecoderNetwork(hidden_dim, projection_dim, n_heads, head_dim, self.shared_dec_layers,
                                              dropout_rate, dropatt_rate, pre_lnorm, same_lengths, pos_enc,
                                              **self.keyword_dict())

        self.language_wise_encoders = nn.ModuleList([
            EncoderNetwork(hidden_dim, projection_dim, n_heads, head_dim, self.specific_enc_layers,
                           dropout_rate, dropatt_rate, pre_lnorm, same_lengths, pos_enc, False)
            for _ in range(num_src_lang)])

        self.language_wise_decoders = nn.ModuleList([
            DecoderNetwork(hidden_dim, projection_dim, n_heads, head_dim, self.specific_dec_layers,
                           dropout_rate, dropatt_rate, pre_lnorm, same_lengths, pos_enc, False)
            for _ in range(num_tgt_lang)])
        self.final = nn.Linear(hidden_dim, vocab_size)

    def encode_src(self, inp):
        src, src_len, src_lang = inp['src'], inp['src_len'], inp['src_lang']
        src_lang = src_lang[0]  # brute-force implementation
        src_mask = self.shared_encoders.get_mask(None, src_len)
        enc_out, _, enc_self_atts = self.shared_encoders(src, None, src_mask)
        enc_out, _, enc_self_atts2 = self.language_wise_encoders[src_lang](enc_out, None, src_mask)
        inp['enc_out'] = enc_out
        inp['enc_self_att'] = enc_self_atts + enc_self_atts2
        return enc_out

    def forward(self, inp):
        src, tgt, src_len, tgt_len, tgt_lang, = \
            inp['src'], inp['tgt'], inp['src_len'], inp['tgt_len'], inp['tgt_lang']

        tgt_lang = tgt_lang[0]  # brute-force implementation
        enc_out = inp['enc_out'] if 'enc_out' in inp else None  # if src is already encoded. used in decoding phase
        tgt_mem = inp['tgt_mem'] if 'tgt_mem' in inp else None
        tgt_mem_shared = tgt_mem[:self.shared_dec_layers] if tgt_mem else None
        tgt_mem_specific = tgt_mem[self.shared_dec_layers:] if tgt_mem else None
        out = {}
        if enc_out is None:
            enc_out = self.encode_src(inp)
        tgt_mask = self.shared_decoders.get_mask(tgt_mem, tgt_len)
        tgt_to_src_mask = self.shared_decoders.tgt_to_src_mask(src_len, tgt_len)
        dec_out, new_tgt_mem, dec_self_att, inter_att = self.shared_decoders(
            enc_out, tgt, tgt_mem_shared, tgt_mask, tgt_to_src_mask)
        dec_out, new_tgt_mem2, dec_self_att2, inter_att2 = self.language_wise_decoders[tgt_lang](
            enc_out, dec_out, tgt_mem_specific, tgt_mask, tgt_to_src_mask)

        logits = self.final(dec_out)
        out['logits'] = logits
        out['tgt_mem'] = new_tgt_mem + new_tgt_mem2
        out['dec_self_att'] = dec_self_att + dec_self_att2
        out['inter_att'] = inter_att + inter_att2
        return out


class ComplexityAwareModel(EncoderDecoderModel):
    def __init__(self, vocab_size: int, seq_len: int, hidden_dim: int, projection_dim: int, n_heads: int, head_dim: int,
                 enc_num_layers: int, dec_num_layers: int, dropout_rate: float, dropatt_rate: float, padding_index: int,
                 cutoffs: list, pre_lnorm: bool = False, same_lengths: bool = False, rel_att: bool = False,
                 shared_embedding=False, tie_embedding=False, **kwargs):
        super(ComplexityAwareModel, self).__init__(vocab_size, seq_len, hidden_dim, projection_dim, n_heads,
                                                   head_dim, enc_num_layers, dec_num_layers, dropout_rate,
                                                   dropatt_rate, padding_index, pre_lnorm, same_lengths,
                                                   rel_att, shared_embedding, tie_embedding, **kwargs)
        delattr(self, 'final')
        self.n_clusters = len(cutoffs) + 1
        self.final = ComplexityControllingSoftmax(vocab_size, hidden_dim, cutoffs, padding_index)
        self.cluster_classification = nn.Linear(hidden_dim, 5)

    def cluster_predict(self, inp):
        enc_out = inp['enc_out']
        cluster_logits = self.cluster_classification(enc_out[:, 0])
        return cluster_logits

    def forward(self, inp):
        src, tgt, src_len, tgt_len, tgt_cluster = \
            inp['src'], inp['tgt'], inp['src_len'], inp['tgt_len'], inp['tgt_cluster']
        enc_out = inp['enc_out'] if 'enc_out' in inp else None  # if src is already encoded. used in decoding phase
        tgt_mem = inp['tgt_mem'] if 'tgt_mem' in inp else None
        out = {}
        if enc_out is None:
            enc_out = self.encode_src(inp)
        tgt_mask = self.decoder.get_mask(tgt_mem, tgt_len)
        tgt_to_src_mask = self.decoder.tgt_to_src_mask(src_len, tgt_len)
        dec_out, new_tgt_mem, dec_self_att, inter_att = self.decoder(enc_out, tgt, tgt_mem, tgt_mask, tgt_to_src_mask)
        cluster_logits = self.cluster_classification(enc_out[:, 0])
        logits = self.final(dec_out, tgt_cluster)
        out['logits'] = logits
        out['cluster_logits'] = cluster_logits
        out['tgt_mem'] = new_tgt_mem
        out['dec_self_att'] = dec_self_att
        out['inter_att'] = inter_att
        return out


class SentenceAwareModel(EncoderDecoderModel):
    def __init__(self, vocab_size: int, seq_len: int, hidden_dim: int, projection_dim: int, n_heads: int, head_dim: int,
                 enc_num_layers: int, dec_num_layers: int, dropout_rate: float, dropatt_rate: float, padding_index: int,
                 pre_lnorm: bool = False, same_lengths: bool = False, pos_enc: str = 'absolute', shared_embedding=False,
                 tie_embedding=False, **kwargs):
        super(SentenceAwareModel, self).__init__(vocab_size, seq_len, hidden_dim, projection_dim, n_heads,
                                                 head_dim, enc_num_layers, dec_num_layers, dropout_rate,
                                                 dropatt_rate, padding_index, pre_lnorm, same_lengths,
                                                 pos_enc, shared_embedding, tie_embedding, **kwargs)
        self.tgt_emb_prediction = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.ReLU(),
                                                nn.Linear(hidden_dim * 4, hidden_dim),
                                                nn.LayerNorm(hidden_dim, elementwise_affine=False))
        self.tgt_encoder = TextConv(hidden_dim, hidden_dim // 4,
                                    dropout=dropout_rate, padding_idx=padding_index, **self.keyword_dict())
        # self.tgt_encoder = EncoderNetwork(hidden_dim, projection_dim, n_heads, head_dim, enc_num_layers, dropout_rate,
        #                                   dropatt_rate, pre_lnorm, same_lengths, rel_att,
        #                                   vocab_size=vocab_size, seq_len=seq_len, padding_index=padding_index)
        # self.tgt_emb_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def build_decoder_network(self, decoder_type=SentenceAwareDecoderBlock):
        return super().build_decoder_network(block_type=decoder_type)

    def stochastic_tgt_emb(self, tgt_emb_out):
        avg, dev = tgt_emb_out.chunk(2, dim=-1)
        res = avg + torch.rand_like(dev) * dev
        return res

    def encode_src(self, inp):
        enc_out = super().encode_src(inp)
        tgt_emb_hat = self.tgt_emb_prediction(enc_out[:, 0])
        # tgt_emb_hat = self.stochastic_tgt_emb(self.tgt_emb_prediction(enc_out[:, 0]))
        inp['tgt_emb_hat'] = tgt_emb_hat
        return enc_out, tgt_emb_hat

    def encode_tgt(self, inp):
        tgt, tgt_len = inp['tgt'], inp['tgt_len']

        # if tgt_encoder is transformer
        # tgt_mask = self.tgt_encoder.get_mask(None, tgt_len)
        # enc_out, _, enc_self_atts = self.tgt_encoder(tgt, None, tgt_mask)

        enc_out = self.tgt_encoder(tgt)
        return enc_out

    def forward(self, inp):
        src, tgt, src_len, tgt_len = inp['src'], inp['tgt'], inp['src_len'], inp['tgt_len']
        is_sampling = not self.training and 'enc_out' in inp
        enc_out = inp['enc_out'] if 'enc_out' in inp else None  # if src is already encoded. used in decoding phase
        tgt_emb_hat = inp['tgt_emb_hat'] if 'tgt_emb_hat' in inp else None
        tgt_mem = inp['tgt_mem'] if 'tgt_mem' in inp else None
        out = {}
        if enc_out is None:
            enc_out, tgt_emb_hat = self.encode_src(inp)
        if is_sampling:
            tgt_emb = tgt_emb_hat
        else:
            tgt_emb_out = self.encode_tgt(inp)
            out['tgt_emb'] = tgt_emb_out
            if self.training:
                tgt_emb = tgt_emb_out
            else:
                tgt_emb = tgt_emb_hat
        tgt_mask = self.decoder.get_mask(tgt_mem, tgt_len)
        tgt_to_src_mask = self.decoder.tgt_to_src_mask(src_len, tgt_len)
        tgt_emb_extended = tgt_emb.detach().unsqueeze(1).repeat(1, tgt.size(1), 1)
        dec_out, new_tgt_mem, dec_self_att, inter_att = self.decoder(enc_out, tgt, tgt_mem, tgt_mask, tgt_to_src_mask,
                                                                     tgt_emb_extended)
        logits = self.final(dec_out)
        out['logits'] = logits
        out['tgt_mem'] = new_tgt_mem
        out['tgt_emb_hat'] = tgt_emb_hat
        out['dec_self_att'] = dec_self_att
        out['inter_att'] = inter_att
        return out
