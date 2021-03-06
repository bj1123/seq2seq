import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ops import pad_input


class TextConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernal_sizes=[1, 3, 4, 5], dropout=0.1, padding_idx= -1,
                 **kwargs):
        super(TextConv, self).__init__()
        self.max_kernal = max(kernal_sizes)
        self.vocab_size = kwargs.get('vocab_size', None)
        self.emb = kwargs.get('embedding', None)  # is passed if encoder and decoder share embedding
        if self.emb is None:
            self.emb = nn.Embedding(self.vocab_size, input_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList([nn.Conv2d(1, output_dim, (K, input_dim)) for K in kernal_sizes])
        self.ln = nn.LayerNorm(len(kernal_sizes) * output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.emb(x, None)
        x = pad_input(x, self.max_kernal)
        x = x[:,None]
        res = [i(x).squeeze(3) for i in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in res]
        x = torch.cat(x, 1)
        x = self.ln(x)
        x = self.dropout(x)
        return x
