import torch
from models.transformer.transformer import Transformer
from utils.data_utils import get_input_data

def infer(model, input_data, device):
    model.eval()
    with torch.no_grad():
        src = torch.tensor(input_data).to(device)
        output = model(src, src)  # Example: using src as both input and target for simplicity
    return output

def main():
    # Load model
    model = Transformer(input_dim=10000, output_dim=10000, emb_dim=512, n_heads=8, n_layers=6, pf_dim=2048, dropout=0.1)

    # Load model checkpoint
    model.load_state_dict(torch.load('training/checkpoints/model_checkpoint.pth'))
    print('Checkpoint loaded')

    # Inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    input_data = get_input_data()
    output = infer(model, input_data, device)
    print(f'Inference result: {output}')

if __name__ == '__main__':
    main()