# src/recommenders.py
from typing import List
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import joblib

class ContentRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=1)
        self.tfidf_matrix = None
        self.movies = None

    def fit(self, movies_df: pd.DataFrame):
        # movies_df must have ['movieId','title','genres']
        self.movies = movies_df.reset_index(drop=True)
        corpus = self.movies['genres'].fillna('') + ' ' + self.movies['title'].fillna('')
        self.tfidf_matrix = self.tfidf.fit_transform(corpus)

    def recommend(self, movie_title: str, topn: int = 10) -> List[dict]:
        idx = self.movies[self.movies['title'].str.contains(movie_title, case=False, na=False)].index
        if len(idx)==0:
            return []
        i = idx[0]
        cosine_similarities = linear_kernel(self.tfidf_matrix[i:i+1], self.tfidf_matrix).flatten()
        related_idx = cosine_similarities.argsort()[:-topn-1:-1]
        results = self.movies.iloc[related_idx][['movieId','title']]
        return results.to_dict(orient='records')

class CollaborativeRecommenderFallback:
    """
    A simple item-item collaborative approach using TruncatedSVD on user-item matrix,
    then nearest neighbors in latent space.
    """
    def __init__(self, n_components=50):
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.nn = NearestNeighbors(n_neighbors=11, metric='cosine')
        self.movie_ids = None
        self.latent = None

    def fit(self, ratings_df: pd.DataFrame):
        # ratings_df: userId, movieId, rating
        pivot = ratings_df.pivot_table(index='movieId', columns='userId', values='rating', fill_value=0)
        self.movie_ids = pivot.index.values
        mat = pivot.values
        self.latent = self.svd.fit_transform(mat)
        self.nn.fit(self.latent)

    def recommend(self, movie_id:int, topn=10):
        if self.latent is None:
            return []
        try:
            idx = int(np.where(self.movie_ids==movie_id)[0][0])
        except IndexError:
            return []
        dist, ind = self.nn.kneighbors(self.latent[idx:idx+1], n_neighbors=topn+1)
        rec_ids = self.movie_ids[ind.flatten()[1:]]
        return rec_ids.tolist()

# Save/load helpers
def save_model(obj, path):
    joblib.dump(obj, path)

def load_model(path):
    return joblib.load(path)
