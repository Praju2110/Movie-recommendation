# Movie Recommendation System — Complete Python Project

This repository contains a complete, runnable movie recommendation system in Python using basic ML libraries.

Features:
- Content-based recommender (TF-IDF on genres + title + cosine similarity)
- Collaborative fallback (TruncatedSVD + NearestNeighbors)
- Scripts for downloading and preprocessing MovieLens data (ml-100k)
- Minimal Flask app to try recommendations locally
- Tests and requirements

Quick start:
1. Create venv and install:
   ```
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Download MovieLens 100k into `data/`:
   ```
   python scripts/download_movielens.py --dataset ml-100k --out data/
   ```
3. Preprocess:
   ```
   python scripts/preprocess.py --data-dir data/ --out-dir data/processed
   ```
4. Run app:
   ```
   python src/app.py
   # open http://127.0.0.1:5000
   ```
