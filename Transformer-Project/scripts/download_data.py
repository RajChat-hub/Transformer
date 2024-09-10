import requests

def download_data(url, output_path):
    response = requests.get(url)
    with open(output_path, 'wb') as file:
        file.write(response.content)

if __name__ == '__main__':
    url = 'https://example.com/dataset.csv'
    output_path = 'data/raw/dataset.csv'
    download_data(url, output_path)