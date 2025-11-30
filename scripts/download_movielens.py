# scripts/download_movielens.py
import argparse
import os
import requests
import zipfile
from io import BytesIO

URLS = {
    'ml-100k': 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'ml-1m': 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
}

def download(dataset: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    url = URLS.get(dataset)
    if not url:
        raise ValueError('Unknown dataset')
    print(f'Downloading {dataset} from {url} ...')
    r = requests.get(url, stream=True)
    r.raise_for_status()
    z = zipfile.ZipFile(BytesIO(r.content))
    z.extractall(out_dir)
    print(f'Extracted to {out_dir}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-100k')
    p.add_argument('--out', dest='out', default='data/')
    args = p.parse_args()
    download(args.dataset, args.out)
