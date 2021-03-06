import torch
import torch.nn as nn
from util.sampler import top_p_logits, top_k_logits
from model.ops import gelu


class AdaptiveBase(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, cutoffs=None, div_val=2):
        super(AdaptiveBase, self).__init__()
        self.vocab_size = vocab_size
        self.n_cluster = len(cutoffs) + 1 if cutoffs is not None else 4
        self.projection_dim = hidden_dim
        self.scale = hidden_dim ** 0.5
        cutoffs = cutoffs if cutoffs else self.compute_cutoffs()
        self.cutoffs = [0] + cutoffs + [vocab_size]
        self.embedding_dims = [hidden_dim // (div_val ** i) for i in range(self.n_cluster)]

    def compute_cutoffs(self):
        target_ratio = [0.05, 0.2, 0.5]
        return [int(i * self.vocab_size) for i in target_ratio]


class SemiAdaptiveSoftmax(AdaptiveBase):
    def __init__(self, vocab_size: int, hidden_dim: int,
                 cutoffs=None, div_val=2):
        super(SemiAdaptiveSoftmax, self).__init__(vocab_size, hidden_dim, cutoffs, div_val)
        self.projections = nn.ModuleList()
        self.tails = nn.ModuleList()
        self.scales = nn.ModuleList()
        self.global_scale = nn.Parameter(torch.Tensor([1.0]))
        for i in range(self.n_cluster):
            n_vocabs = self.cutoffs[i + 1] - self.cutoffs[i]
            self.projections.append(
                    nn.Linear(hidden_dim, self.embedding_dims[i]))
            self.tails.append(nn.Linear(self.embedding_dims[i], n_vocabs, bias=False))
            self.scales.append(nn.LayerNorm(n_vocabs, elementwise_affine=False))

    def forward(self, x):
        tail_word_logits = []
        for i in range(self.n_cluster):
            proj = self.projections[i](x)
            tail = self.tails[i](proj)
            tail = self.scales[i](tail)
            tail_word_logits.append(tail)
        return torch.cat(tail_word_logits, -1) * self.global_scale


class AdaptiveSoftmax(AdaptiveBase):
    def __init__(self, vocab_size: int, hidden_dim: int,
                 cutoffs=None, div_val=2):
        super(AdaptiveSoftmax, self).__init__(vocab_size, hidden_dim, cutoffs, div_val)
        self.head = nn.Linear(hidden_dim, self.cutoffs[1] + self.n_cluster - 1, bias=False)
        self.projections = nn.ModuleList()
        self.tails = nn.ModuleList()
        for i in range(1, self.n_cluster):
            n_vocabs = self.cutoffs[i + 1] - self.cutoffs[i]
            self.projections.append(nn.Linear(hidden_dim, self.embedding_dims[i], bias=False))
            self.tails.append(nn.Linear(self.embedding_dims[i], n_vocabs, bias=False))

    def forward(self, x):
        head = self.head(x)
        head_word_logits, cluster_logits = head[..., :self.cutoffs[1]], head[..., self.cutoffs[1]:]
        tail_word_logits = []
        for i in range(self.n_cluster - 1):
            proj = self.projections[i](x)
            tail = self.tails[i](proj)
            tail_word_logits.append(cluster_logits[..., i].unsqueeze(-1) + torch.log_softmax(tail, -1))
        return torch.cat([head_word_logits, *tail_word_logits], -1)


class HashSoftmax(nn.Module):
    def __init__(self, vocab_size, hidden_dim, pool_size=5000, num_hash=20):
        super(HashSoftmax, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pool = nn.Embedding(pool_size, hidden_dim)
        self.import_params = nn.Parameter(torch.Tensor(vocab_size, num_hash))
        nn.init.constant_(self.import_params, 1 / num_hash)
        self.hash_values = nn.Parameter(torch.randint(0, pool_size, (vocab_size, num_hash), dtype=torch.long),
                                        requires_grad=False)

    def forward(self, x):
        if x.is_cuda and not self.hash_values.is_cuda:
            self.hash_values = self.hash_values.cuda()
        hash_values = nn.functional.embedding(torch.arange(self.vocab_size, device=x.device), self.hash_values)
        import_params = nn.functional.embedding(torch.arange(self.vocab_size, device=x.device), self.import_params)
        embed = self.pool(hash_values)  # [vs, nh,  embedding_dim]
        embed = (import_params.unsqueeze(-1) * embed).sum(1)
        return torch.matmul(x, embed.T)


class LinearTransform(nn.Module):
    def __init__(self, hidden_states: int, activation_fn):
        super(LinearTransform, self).__init__()
        self.linear = nn.Linear(hidden_states, hidden_states)
        self.ln = nn.LayerNorm(hidden_states)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.activation_fn(self.linear(x))
        return self.ln(x)
