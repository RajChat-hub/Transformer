import unittest
import torch
from training.scripts.train import train_model
from models.transformer.transformer import Transformer
from utils.data_utils import get_dataloader

class TestTraining(unittest.TestCase):
    def test_training_loop(self):
        dataloader = get_dataloader()
        model = Transformer(input_dim=10000, output_dim=10000, emb_dim=512, n_heads=8, n_layers=6, pf_dim=2048, dropout=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = torch.nn.CrossEntropyLoss()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        try:
            train_model(model, dataloader, optimizer, criterion, device)
            success = True
        except Exception:
            success = False
        
        self.assertTrue(success)

if __name__ == '__main__':
    unittest.main()