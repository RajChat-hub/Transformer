import sys
import torch
import torch.optim as optim
from models.transformer.transformer import Transformer
from utils.data_utils import get_dataloader

# Print Python path for debugging
print("Python Path:")
for path in sys.path:
    print(path)

def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    for batch in dataloader:
        src, tgt = batch
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        output = model(src, tgt)
        loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        loss.backward()
        optimizer.step()
        
    print('Training complete')

def main():
    # Load data
    dataloader = get_dataloader()

    # Model and optimizer
    model = Transformer(input_dim=10000, output_dim=10000, emb_dim=512, n_heads=8, n_layers=6, pf_dim=2048, dropout=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    # Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_model(model, dataloader, optimizer, criterion, device)

if __name__ == '__main__':
    main()
