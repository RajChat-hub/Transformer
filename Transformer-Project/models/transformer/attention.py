import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // n_heads
        assert self.head_dim * n_heads == emb_dim, "Embedding dimension must be divisible by number of heads"

        self.q_linear = nn.Linear(emb_dim, emb_dim)
        self.k_linear = nn.Linear(emb_dim, emb_dim)
        self.v_linear = nn.Linear(emb_dim, emb_dim)
        self.fc_out = nn.Linear(emb_dim, emb_dim)

    def forward(self, query, key, value, mask=None):
        N = query.shape[0]
        Q = self.q_linear(query).view(N, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(N, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(N, -1, self.n_heads, self.head_dim).transpose(1, 2)

        energy = torch.matmul(Q, K.transpose(-1, -2))
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(energy / (self.emb_dim ** (1 / 2)), dim=-1)

        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(N, -1, self.emb_dim)
        out = self.fc_out(out)
        return out