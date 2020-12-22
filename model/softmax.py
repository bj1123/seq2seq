import torch
import torch.nn as nn
from util.sampler import top_p_logits, top_k_logits
from model.ops import gelu


class AdaptiveSoftmax(nn.Module):
    def __init__(self,vocab_size:int,hidden_dim:int,cutoffs:list,div_val:int):
        super(AdaptiveSoftmax, self).__init__()
        self.n_clusters = len(cutoffs)
        self.head_size = cutoffs[0] + self.n_clusters
        self.cutoffs = [0] + cutoffs + [vocab_size]

        self.cluster_logit = nn.Linear(hidden_dim,self.n_clusters)
        self.head_size = cutoffs[0] + self.n_clusters

        self.projections = nn.ModuleList()
        self.logits = nn.ModuleList()
        self.proj_dims = [hidden_dim // (div_val**i) for i in range(self.n_clusters+1)]
        for i in range(self.n_clusters+1):
            n_vocabs = self.cutoffs[i] + self.cutoffs[i+1]
            self.projections.append(nn.Linear(hidden_dim,self.proj_dims[i],bias=False))
            self.logits.append(nn.Linear(self.proj_dims[i],n_vocabs))

    def forward(self, x,y):
        """
        :param x: final hidden state x.size() = [batch_size*seq_len,hidden_dim]
        :param y: target y.size() = [batch_size*seq_len]
        :return:
        """
        head_proj = self.projections[0](x)
        head_logit = torch.cat([self.logits[0](head_proj),self.cluster_logit(head_proj)],1)
        head_logprob = torch.log_softmax(head_logit, dim=1)

        nll = torch.zeros_like(y,
                               dtype=x.dtype, device=x.device)

        for i in range(len(self.cutoffs)-1):
            l,r = self.cutoffs[i], self.cutoffs[i+1]
            mask = (y >= l) & (y < r)
            indices = mask.nonzero().squeeze()
            if indices.numel() == 0:
                continue
            target_i = y[indices] - l
            head_logprob_i = head_logprob[indices]
            if i == 0:
                logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
            else:
                tail_proj = self.projections[i](x[indices])
                tail_logit = self.logits[i](tail_proj)
                tail_logprob_i = torch.log_softmax(tail_logit, dim=1)
                logprob_i = head_logprob_i[:, -i] \
                            + tail_logprob_i.gather(1, target_i[:, None]).squeeze(1)
            nll[indices] = -logprob_i
        return nll


class FactorizedSoftmax(nn.Module):
    def __init__(self,vocab_size:int,hidden_dim:int,cutoffs:list,padding_index:int, activation=gelu):
        super(FactorizedSoftmax, self).__init__()
        self.padding_index = padding_index
        self.vocab_size = vocab_size
        self.n_clusters = len(cutoffs) + 1
        self.cutoffs = [0] + cutoffs + [vocab_size]
        self.cluster_logit = nn.Linear(hidden_dim, self.n_clusters, bias=False)
        self.logits = nn.Parameter(torch.Tensor(hidden_dim, vocab_size))
        self.transform = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        self.activation=activation
        for i in range(self.n_clusters):
            self.transform.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim)))
            self.layer_norm.append(nn.LayerNorm(hidden_dim))

    def hard_cluster_logit(self,x, top_w, ishard=True):
        def check_shorts(logits, top_w):
            if isinstance(top_w,int): #if case is top_k
                # print(((logits != 0).sum(dim=1) < top_w).sum())
                res = (logits != 0).sum(dim=1) < top_w
            elif isinstance(top_w,float): #if case is top_p
                res = logits.sum(dim=1) < top_w
            else:
                raise TypeError('type of top_w should be either int or float')
            return res

        logits = torch.zeros(x.size(0),self.vocab_size).to(x.device)
        cl = self.cluster_logit(x)
        cl_probs = torch.softmax(cl,1)

        if ishard:
            _, target_cluster = torch.topk(cl,self.n_clusters, dim=1)
        else:
            cl = top_p_logits(cl,0.6)
            target_cluster = torch.multinomial(torch.softmax(cl,1) + 1e-6, self.n_clusters)
        idx = 0
        while True:
            cs = check_shorts(logits,top_w)
            if cs.sum() == 0:
                break
            for i in range(self.n_clusters):
                l,r = self.cutoffs[i], self.cutoffs[i+1]
                indices = ((target_cluster[:,idx] == i) & cs).nonzero().squeeze(1)
                transformed = self.layer_norm[i](self.activation(self.transform[i](x[indices])))
                tail = torch.softmax(torch.matmul(transformed, self.logits[:,l:r]),1)
                logits[indices,l:r] = cl_probs[indices,i].unsqueeze(1) * tail
            idx+=1
        return torch.log(logits)

    def soft_cluster_logit(self,x):
        logits = torch.zeros(x.size(0), self.vocab_size).to(x.device)
        cl = self.cluster_logit(x)
        cluster_prob = torch.softmax(cl,dim=1) # [ batch, n_cluster]
        for i in range(self.n_clusters):
            l,r = self.cutoffs[i], self.cutoffs[i+1]
            logits_weights = self.logits[:,l:r]
            transformed = self.layer_norm[i](self.activation(self.transform[i](x)))
            tail_logit = torch.matmul(transformed,logits_weights)
            tail_prob = torch.softmax(tail_logit,1)
            # print(cluster_prob[:,i].size(),tail_prob.size())
            logits[:,l:r] = cluster_prob[:,i].unsqueeze(1) * tail_prob
        return torch.log(logits)

    def forward(self, x,y):
        """
        :param x: final hidden state x.size() = [batch_size*seq_len,hidden_dim]
        :param y: target y.size() = [batch_size*seq_len]
        :return:
        """
        ny = y.size(0)
        cl = self.cluster_logit(x)
        cluster_ll = torch.log_softmax(cl, dim=1)
        nll = torch.zeros_like(y,
                               dtype=x.dtype, device=x.device)

        for i in range(self.n_clusters):
            l,r = self.cutoffs[i], self.cutoffs[i+1]
            mask = (y >= l) & (y < r)
            indices = mask.nonzero().squeeze(1)
            logits_weights = self.logits[:,l:r]
            if indices.numel() == 0:
                continue
            target_i = y[indices] - l
            transformed = self.layer_norm[i](self.activation(self.transform[i](x[indices])))
            tail_logit = torch.matmul(transformed,logits_weights)
            tail_logprob_i = torch.log_softmax(tail_logit, dim=1) # [b,vocab]
            # word_nll[indices] = -logprob_i
            nll[indices] = - cluster_ll[indices, i] - tail_logprob_i.gather(1,target_i[:,None]).squeeze(1)
        padding_mask = y == self.padding_index
        padding_indices = padding_mask.nonzero().squeeze(1)
        padding_size = padding_indices.size(0)
        nll[padding_indices] = 0
        return torch.sum(nll) / (ny-padding_size)


class FactorizedSoftmaxV2(nn.Module):
    def __init__(self,vocab_size:int,hidden_dim:int,cutoffs:list,padding_index:int, pretrained=None, **kwargs):
        super(FactorizedSoftmaxV2, self).__init__()
        self.padding_index = padding_index
        self.vocab_size = vocab_size
        self.n_clusters = len(cutoffs) + 1
        self.cutoffs = [0] + cutoffs + [vocab_size]
        self.cluster_logit = nn.Linear(hidden_dim, self.n_clusters, bias=False)
        if pretrained is not None:
            self.logits = nn.Parameter(pretrained)
        else:
            self.logits = nn.Parameter(torch.Tensor(hidden_dim, vocab_size))

        self.init_weight()

    def init_weight(self):
        torch.nn.init.normal_(self.cluster_logit.weight, std=0.02)

    def hard_cluster_logit(self,x, top_w, ishard=True, temperature=1.0):
        def check_shorts(logits, top_w):
            if isinstance(top_w,int): #if case is top_k
                # print(((logits != 0).sum(dim=1) < top_w).sum())
                # res = (logits != 0).sum(dim=1) < top_w
                res = filled < top_w
            elif isinstance(top_w,float): #if case is top_p
                res = logits.sum(dim=1) < top_w
            else:
                raise TypeError('type of top_w should be either int or float')
            return res

        logits = torch.zeros(x.size(0),self.vocab_size, dtype=x.dtype, device=x.device)
        filled = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        cl = self.cluster_logit(x)
        cl_probs = torch.softmax(cl,1)
        if ishard:
            _, target_cluster = torch.topk(cl,self.n_clusters, dim=1)
        else:
            # cl = top_k_logits(cl,3)
            cl = top_p_logits(cl,0.9)
            target_cluster = torch.multinomial(torch.softmax(cl,1) + 1e-6, self.n_clusters)
        idx = 0
        while True:
            cs = check_shorts(logits,top_w)
            if cs.sum() == 0:
                break
            for i in range(self.n_clusters):
                l,r = self.cutoffs[i], self.cutoffs[i+1]
                indices = ((target_cluster[:,idx] == i) & cs).nonzero().squeeze(1)
                tail = torch.softmax(torch.matmul(x[indices], self.logits[:,l:r]),1)
                if isinstance(top_w, float) and ishard==False:
                    logits[indices, l:r] = tail
                else:
                    # logits[indices, l:r] = tail
                    logits[indices, l:r] = cl_probs[indices, i].unsqueeze(1) * tail / 0.9
                    filled[indices] += tail.size(1)
            idx+=1
        return torch.log(logits)

    def soft_cluster_logit(self,x):
        logits = torch.zeros(x.size(0), self.vocab_size).to(x.device)
        cl = self.cluster_logit(x)
        cluster_prob = torch.softmax(cl,dim=1) # [ batch, n_cluster]
        for i in range(self.n_clusters):
            l,r = self.cutoffs[i], self.cutoffs[i+1]
            tail_prob = torch.softmax(torch.matmul(x, self.logits[:,l:r]),1)
            # print(cluster_prob[:,i].size(),tail_prob.size())
            logits[:,l:r] = cluster_prob[:,i].unsqueeze(1) * tail_prob
        return torch.log(logits)

    def forward(self, x,y):
        """
        :param x: final hidden state x.size() = [batch_size*seq_len,hidden_dim]
        :param y: target y.size() = [batch_size*seq_len]
        :return:
        """
        ny = y.size(0)
        cl = self.cluster_logit(x)
        cluster_ll = torch.log_softmax(cl, dim=1)
        nll = torch.zeros_like(y,
                               dtype=x.dtype, device=x.device)

        for i in range(self.n_clusters):
            l,r = self.cutoffs[i], self.cutoffs[i+1]
            mask = (y >= l) & (y < r)
            indices = mask.nonzero().squeeze(1)
            logits_weights = self.logits[:,l:r]
            if indices.numel() == 0:
                continue
            target_i = y[indices] - l
            tail_logit = torch.matmul(x[indices],logits_weights)
            tail_logprob_i = torch.log_softmax(tail_logit, dim=1) # [b,vocab]
            # word_nll[indices] = -logprob_i
            nll[indices] = - cluster_ll[indices, i] - tail_logprob_i.gather(1,target_i[:,None]).squeeze(1)
        return nll
        # return torch.sum(nll) / (ny-padding_size)


class ComplexityControllingSoftmax(nn.Module):
    def __init__(self, vocab_size:int, hidden_dim:int, cutoffs:list, padding_index:int, **kwargs):
        super(ComplexityControllingSoftmax, self).__init__()
        self.padding_index = padding_index
        self.vocab_size = vocab_size
        self.n_clusters = len(cutoffs) + 1
        self.cutoffs = [0] + cutoffs + [vocab_size]
        self.cluster_logit = nn.Linear(hidden_dim, self.n_clusters, bias=False)
        self.cluster_embedding = nn.Embedding(self.n_clusters, hidden_dim)
        self.logits = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x, clusters):
        """
        :param x: [bs, l, dim]
        :param clusters: [bs ]
        :return:
        """
        words_logits = self.logits(x)  # bs, l, vocab_size
        cluster_embeded = self.cluster_embedding(clusters)
        cluster_logits = self.cluster_logit(x+cluster_embeded[:, None])  # bs, l, n_cluster
        logits = torch.zeros(*x.size()[:2], self.vocab_size, dtype=x.dtype, device=x.device)
        for i in range(self.n_clusters):
            l, r = self.cutoffs[i], self.cutoffs[i + 1]
            tail_prob = words_logits[..., l:r]
            logits[:, l:r] = cluster_logits[..., i].unsqueeze(-1) + tail_prob
        return logits


class LinearTransform(nn.Module):
    def __init__(self,hidden_states:int,activation_fn):
        super(LinearTransform, self).__init__()
        self.linear = nn.Linear(hidden_states,hidden_states)
        self.ln = nn.LayerNorm(hidden_states)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.activation_fn(self.linear(x))
        return self.ln(x)
