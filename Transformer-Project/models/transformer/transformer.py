import torch
import torch.nn as nn
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, n_heads, n_layers, pf_dim, dropout):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(input_dim, emb_dim, n_heads, n_layers, pf_dim, dropout)
        self.decoder = TransformerDecoder(output_dim, emb_dim, n_heads, n_layers, pf_dim, dropout)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output