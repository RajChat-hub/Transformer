import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, n_heads, n_layers, pf_dim, dropout):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, emb_dim))
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=pf_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(emb_dim, output_dim)

    def forward(self, tgt, memory):
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1)]
        tgt = self.dropout(tgt)
        for layer in self.layers:
            tgt = layer(tgt, memory)
        output = self.fc_out(tgt)
        return output