import torch
from models.transformer.transformer import Transformer
from utils.data_utils import get_dataloader
from utils.metrics import calculate_metrics

def evaluate_model(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        total_metrics = 0
        for batch in dataloader:
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)

            output = model(src, tgt)
            metrics = calculate_metrics(output, tgt)
            total_metrics += metrics

    print('Evaluation complete')
    print(f'Metrics: {total_metrics / len(dataloader)}')

def main():
    # Load data
    dataloader = get_dataloader()

    # Model
    model = Transformer(input_dim=10000, output_dim=10000, emb_dim=512, n_heads=8, n_layers=6, pf_dim=2048, dropout=0.1)

    # Load model checkpoint
    model.load_state_dict(torch.load('training/checkpoints/model_checkpoint.pth'))
    print('Checkpoint loaded')

    # Evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    evaluate_model(model, dataloader, device)

if __name__ == '__main__':
    main()