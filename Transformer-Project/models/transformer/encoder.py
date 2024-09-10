import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, n_heads, n_layers, pf_dim, dropout):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, emb_dim))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=pf_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.embedding(src) + self.positional_encoding[:, :src.size(1)]
        src = self.dropout(src)
        for layer in self.layers:
            src = layer(src)
        return src