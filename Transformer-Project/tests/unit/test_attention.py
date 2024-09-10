import unittest
import torch
from models.transformer.attention import MultiHeadAttention

class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        self.attention = MultiHeadAttention(emb_dim=512, n_heads=8)
        self.query = torch.rand(64, 10, 512)
        self.key = torch.rand(64, 10, 512)
        self.value = torch.rand(64, 10, 512)

    def test_attention_forward(self):
        output, _ = self.attention(self.query, self.key, self.value)
        self.assertEqual(output.shape, self.query.shape)

if __name__ == '__main__':
    unittest.main()