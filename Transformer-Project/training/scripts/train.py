import torch
import torch.optim as optim
from models.transformer.transformer import Transformer
from utils.data_utils import get_dataloader

def train_model(model, dataloader, optimizer, criterion, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')

        # Save checkpoint after each epoch
        checkpoint_path = f'training/checkpoints/model_checkpoint_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')

    # Save final model checkpoint
    torch.save(model.state_dict(), 'training/checkpoints/model_checkpoint.pth')
    print('Final model checkpoint saved')

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