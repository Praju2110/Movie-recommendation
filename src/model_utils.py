# src/model_utils.py
import pandas as pd
import os

def load_processed(data_dir='data/processed'):
    path = os.path.join(data_dir, 'movies.csv')
    return pd.read_csv(path)
