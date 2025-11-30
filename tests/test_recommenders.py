# tests/test_recommenders.py
import pandas as pd
from src.recommenders import ContentRecommender

def test_content_basic():
    df = pd.DataFrame({
        'movieId':[1,2,3],
        'title':['Alpha','Beta','Gamma'],
        'genres':['action adventure','action drama','romance comedy']
    })
    c = ContentRecommender()
    c.fit(df)
    res = c.recommend('Alpha', topn=2)
    assert len(res) >= 1
