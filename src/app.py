# src/app.py
from flask import Flask, request, render_template
import os
import sys

# allow importing local src modules when running this script directly
sys.path.append(os.path.dirname(__file__))

from recommenders import ContentRecommender
import model_utils as mu

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Load processed movies if available
try:
    movies = mu.load_processed('data/processed')
    content = ContentRecommender()
    content.fit(movies)
except Exception as e:
    movies = None
    content = None
    print('Warning: could not load processed movies:', e)

@app.route('/', methods=['GET', 'POST'])
def index():
    query = request.form.get('query') if request.method=='POST' else ''
    results = []
    if query and content is not None:
        results = content.recommend(query, topn=10)
    return render_template('index.html', results=results, query=query)

if __name__ == '__main__':
    app.run(debug=True)
