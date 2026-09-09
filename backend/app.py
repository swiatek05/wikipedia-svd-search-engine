from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.preprocessing import normalize

from config import DB_PATH, K_FACTOR, MAX_DOCS, init_resources
from database.data_fetcher import fetch_articles_from_database, get_article_content
from logic.math_logic import (
    apply_idf,
    apply_svd_LRA,
    build_word_document_matrix,
    search_classic,
    search_semantic,
)
from logic.processed_data import load_processed_data, save_processed_data

app = Flask(__name__)
CORS(app)

init_resources()
data = load_processed_data()
if data:
    titles = data['titles']
    word_to_idx = data['word_to_idx']
    idf_values = data['idf_values']
    A_tfidf_norm = data['A_tfidf_norm']
    u_k = data['u_k']
    low_rank_norm = data['low_rank_norm']
else:
    print("Brak zapisanych macierzy")
    print("Ładowanie danych z bazy")
    articles = fetch_articles_from_database(DB_PATH, MAX_DOCS)

    print("Budowa macierzy")
    A, titles, word_to_idx = build_word_document_matrix(articles)

    print("Uwzględnienie wag IDF")
    A_tfidf, idf_values = apply_idf(A)

    A_tfidf_norm = normalize(A_tfidf, axis=0, norm='l2')

    print("Obliczanie SVD dla k=", K_FACTOR)
    u_k, s_k, low_rank_docs = apply_svd_LRA(A_tfidf, k=K_FACTOR)
    low_rank_norm = normalize(low_rank_docs, axis=0, norm='l2')

    save_processed_data({
        'titles': titles,
        'word_to_idx': word_to_idx,
        'idf_values': idf_values,
        'A_tfidf_norm': A_tfidf_norm,
        'u_k': u_k,
        'low_rank_norm': low_rank_norm
    })

    print("Gotowe")


@app.route("/api/search", methods=["POST"])
def search():
    data = request.json or {}
    query = data.get("query")

    if not query:
        return jsonify({"error": "Puste zapytanie"}), 400

    results_tfidf = search_classic(query, word_to_idx, idf_values, A_tfidf_norm, titles)
    results_svd_LRA = search_semantic(query, word_to_idx, idf_values, u_k, low_rank_norm, titles)

    return jsonify({
        "tfidf": [{"title": t, "score": float(s)} for t, s in results_tfidf],
        "svd_LRA": [{"title": t, "score": float(s)} for t, s in results_svd_LRA]
    })


@app.route("/api/article", methods=["POST"])
def article():
    data = request.json or {}
    title = data.get("title")
    content = get_article_content(DB_PATH, title)
    return jsonify({"content": content})


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)