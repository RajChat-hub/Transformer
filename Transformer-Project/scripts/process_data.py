import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(input_file, output_file):
    data = pd.read_csv(input_file)
    # Example preprocessing steps
    data = data.dropna()
    
    train_data, test_data = train_test_split(data, test_size=0.2)
    train_data.to_csv(output_file, index=False)

if __name__ == '__main__':
    preprocess_data('data/raw/dataset.csv', 'data/processed/preprocessed_data.csv')