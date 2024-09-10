import unittest
import torch
from models.transformer.decoder import Decoder

class TestDecoder(unittest.TestCase):
    def setUp(self):
        self.decoder = Decoder(output_dim=10000, emb_dim=512, n_heads=8, pf_dim=2048, dropout=0.1)
        self.input = torch.rand(64, 10, 512)  # Example input tensor

    def test_decoder_forward(self):
        output = self.decoder(self.input, self.input)
        self.assertEqual(output.shape, self.input.shape)

if __name__ == '__main__':
    unittest.main()