import torch
import torch.nn as nn
from torch.autograd import Function
from .ops import hard_sigm


class StochasticRelu(nn.Module):
    def __init__(self, scale=0):
        super(StochasticRelu, self).__init__()
        self.scale = nn.Parameter(torch.Tensor([scale]))
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.training:
            noise = x * (0.01 * torch.randn_like(x) - self.scale)
        else:
            noise = - x * self.scale
        return (x < 0) * noise + self.relu(x)


class Bound(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        output = x > 0.5
        return output.to(x.dtype)

    @staticmethod
    def backward(ctx, output_grad):
        x = ctx.saved_tensors
        # x_grad = output_grad.clone()
        x_grad = None

        if ctx.needs_input_grad[0]:
            x_grad = output_grad.clone()

        return x_grad


class StaticBoundaryDecision(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, head_dim: int, dropatt_rate: float, noise=0.01):
        super(StaticBoundaryDecision, self).__init__()
        self.n_head = n_heads
        self.head_dim = head_dim
        self.q = nn.Parameter(torch.Tensor(hidden_dim))
        self.ln = nn.LayerNorm(hidden_dim)
        self.k_net = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.att_scale = 1 / (hidden_dim ** 0.5)
        self.out_scale = 1 / (n_heads ** 0.5)
        self.noise = noise
        self.use_noise = True

    def attend(self, key, mask):
        bs, ks, hs = key.size()

        # print(query.size(),key.size(),value.size(),rel.size())
        # reshaping
        # k = key.view(bs, ks, self.n_head, self.head_dim)
        # q = self.q.view(self.n_head, self.head_dim)
        k = key
        q = self.q
        # q = self.ln(self.q)
        att_score = torch.einsum('bqd,d->bq', k, q).unsqueeze(-1)
        att_score.mul_(self.att_scale)

        # attend
        if mask is None:
            mask = torch.ones(ks).byte()
        encoder_mask = mask.bool()
        att_score.masked_fill_(encoder_mask.unsqueeze(-1), -6e4)
        return att_score

    def normalize(self, score, mask):
        """
        :param score:
        :return:
        """
        l = (mask == False).sum(-1)
        score = score.masked_fill(mask, 0)
        ssum = score.sum(-1)
        smean = ssum / l
        return score - smean[:, None]

    def add_noise(self, x):
        if self.training:
            # std = x.std(1, keepdim=True)
            return x + (torch.randn_like(x) * self.noise)
        else:
            return x

    def forward(self, x, mem, att_mask):
        att_mask = att_mask[:,-1]
        if mem is None:
            mem = torch.Tensor().to(x.device).to(x.dtype)
        c = torch.cat([mem, x], 1)

        # projection
        key = self.k_net(x)

        out = self.attend(key, att_mask)  # size = (batch, lens, haed)
        # out = out.sum(2,keepdim=True)
        # out.mul_(self.out_scale)
        # out = out.mean(2)
        # if self.use_noise:
        #     out = self.add_noise(out)
        # out = out > 0.5
        # out = self.normalize(out,padded_mask)
        out.masked_fill_(att_mask.unsqueeze(-1), -6e4)
        # out = out.squeeze(-1)
        # out = hard_sigm(out)
        # out = torch.relu(out)
        out = torch.sigmoid(out)
        # out = relu1(out)
        # out = out.softmax(-1)

        # out = Bound.apply(out)
        ind = out.nonzero(as_tuple=True)
        return ind, out


class BoundaryDecision(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, head_dim: int, dropatt_rate: float, noise=0.01):
        super(BoundaryDecision, self).__init__()
        self.n_head = n_heads
        self.head_dim = head_dim
        self.qk_net = nn.Linear(hidden_dim, 2 * n_heads * head_dim, bias=False)
        self.v_net = nn.Linear(hidden_dim, n_heads, bias=False)
        self.dropatt = nn.Dropout(dropatt_rate)
        self.att_scale = 1 / (head_dim ** 0.5)
        self.out_scale = 1 / (n_heads ** 0.5)
        self.noise = noise
        self.use_noise = True

    def attend(self, query, key, value, mask):
        bs, qs, hs = query.size()
        ks = key.size(1)
        ms = ks - qs

        # print(query.size(),key.size(),value.size(),rel.size())
        # reshaping
        k = key.view(bs, ks, self.n_head, self.head_dim)
        v = value.view(bs, ks, self.n_head, -1)
        q = query.view(bs, qs, self.n_head, self.head_dim)

        att_score = torch.einsum('bqnd,bknd->bqkn', q, k)
        att_score.mul_(self.att_scale)

        # attend
        if mask is None:
            mask = torch.ones((qs, ks)).byte()
            mask = mask.triu(1 + ms) == 0
        encoder_mask = mask.bool()
        att_score.masked_fill_(encoder_mask.unsqueeze(-1), -6e4)
        # print(att_score)
        att_prob = torch.softmax(att_score, 2)
        att_prob = self.dropatt(att_prob)

        attended = torch.einsum('bqkn,bknd->bqnd', att_prob, v)
        return attended

    def normalize(self, score, mask):
        """
        :param score:
        :return:
        """
        l = (mask == False).sum(-1)
        score = score.masked_fill(mask, 0)
        ssum = score.sum(-1)
        smean = ssum / l
        return score - smean[:, None]

    def add_noise(self, x):
        if self.training:
            # std = x.std(1, keepdim=True)
            return x + (torch.randn_like(x) * self.noise)
        else:
            return x

    def forward(self, x, mem, att_mask):
        if mem is None:
            mem = torch.Tensor().to(x.device).to(x.dtype)
        c = torch.cat([mem, x], 1)

        # projection
        qk = self.qk_net(c)
        query, key = qk.chunk(2, -1)
        # query, key = c, c
        value = self.v_net(x)

        out = self.attend(query, key, value, att_mask)  # size = (batch, query, haed, 1)
        out = out.sum(2)
        # out.mul_(self.out_scale)
        # out = out.mean(2)
        # out = out.sum(2).sum(2, keepdim=True)
        # out = self.add_noise(out)
        padded_mask = att_mask[:, -1]
        # if self.use_noise:
        #     out = self.add_noise(out)
        # out = out > 0.5
        # out = self.normalize(out,padded_mask)
        out.masked_fill_(padded_mask.unsqueeze(-1), -6e4)
        # out = out.squeeze(-1)
        # out = hard_sigm(out)
        # out = torch.relu(out)
        out = torch.sigmoid(out)
        # out = relu1(out)
        # out = out.softmax(-1)

        # out = Bound.apply(out)
        ind = out.nonzero(as_tuple=True)
        return ind, out


class ResidualFF(nn.Module):
    def __init__(self, hidden_dim: int, projection_dim: int, dropout: float, pre_lnorm=False):
        super(ResidualFF, self).__init__()

        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.dropout = dropout

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.pre_lnorm = pre_lnorm

    def forward(self, x):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            out = self.net(self.layer_norm(x))

            ##### residual connection
            output = out + x
        else:
            ##### positionwise feed-forward
            out = self.net(x)

            ##### residual connection + layer normalization
            output = self.layer_norm(x + out)

        return output
