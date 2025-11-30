# scripts/preprocess.py
import argparse
import os
import pandas as pd
import json

def load_ml_100k(path):
    item_path = os.path.join(path, 'ml-100k', 'u.item')
    # columns: movie id | movie title | release date | video release date | IMDb URL | 19 genre flags
    cols = ['movieId', 'title', 'release_date', 'video_release', 'IMDb_URL'] + [f'genre_{i}' for i in range(19)]
    movies = pd.read_csv(item_path, sep='|', encoding='latin-1', names=cols, usecols=range(24))
    genre_cols = [c for c in movies.columns if c.startswith('genre_')]
    def combine_genres(row):
        # Convert binary flags to a space-separated genre token list (we'll keep indices to keep simple)
        genres = [str(i) for i, c in enumerate(genre_cols) if row[c] == 1]
        return ' '.join(genres)
    movies['genres'] = movies.apply(lambda r: ' '.join([str(i) for i,g in zip(range(len(genre_cols)), r[genre_cols]) if g==1]), axis=1)
    # keep movieId numeric
    movies = movies.rename(columns={'movieId': 'movieId', 'title': 'title'})
    return movies[['movieId', 'title', 'genres']]

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/')
    p.add_argument('--out-dir', default='data/processed')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    movies = load_ml_100k(args.data_dir)
    movies.to_csv(os.path.join(args.out_dir, 'movies.csv'), index=False)
    print('Saved processed movies to', os.path.join(args.out_dir, 'movies.csv'))
