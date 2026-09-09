from collections import Counter
from typing import Dict, List, Tuple
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from .text_preprocessing import clean_text


def build_word_document_matrix(
    articles: List[Tuple[str, str]]
) -> Tuple[csr_matrix, List[str], Dict[str, int]]:
    """
    Buduje rzadką macierz częstości występowania słów w dokumentach (term-document matrix).

    Zwraca:
        matrix: rzadka macierz CSR o wymiarach (liczba_słów, liczba_dokumentów),
        titles: lista tytułów dokumentów odpowiadająca kolumnom macierzy,
        word_to_idx: słownik mapujący słowa na indeksy wierszy macierzy.
    """
    rows, columns, data = [], [], []
    titles: List[str] = []
    vocabulary_counter = Counter()
    docs_tokens: List[List[str]] = []

    for title, content in articles:
        tokens = clean_text(content)
        if tokens:
            docs_tokens.append(tokens)
            vocabulary_counter.update(tokens)
            titles.append(title)

    # Filtrujemy słowa występujące sporadycznie (mniej niż 4 razy w całym korpusie)
    filtered_words = [word for word, count in vocabulary_counter.items() if count > 3]
    word_to_idx = {word: i for i, word in enumerate(filtered_words)}

    for doc_idx, tokens in enumerate(docs_tokens):
        local_counter = Counter(tokens)
        for word, count in local_counter.items():
            if word in word_to_idx:
                rows.append(word_to_idx[word])
                columns.append(doc_idx)
                data.append(count)

    matrix = csr_matrix(
        (data, (rows, columns)),
        shape=(len(word_to_idx), len(titles))
    )
    return matrix, titles, word_to_idx


def apply_idf(matrix: csr_matrix) -> Tuple[csr_matrix, np.ndarray]:
    """
    Oblicza wagi i przemnaża przez nie wiersze macierzy częstości.

    Wzór: IDF_t = ln(N / (df_t + eps))
    """
    num_docs = matrix.shape[1]
    nw = np.diff(matrix.indptr)  # liczba dokumentów, w których występuje dany term

    idf = np.log(num_docs / (nw + 1e-9))
    weight = np.repeat(idf, nw)
    idf_data = matrix.data * weight

    idf_matrix = csr_matrix(
        (idf_data, matrix.indices, matrix.indptr),
        shape=matrix.shape
    )
    return idf_matrix, idf


def apply_svd_LRA(
    matrix: csr_matrix,
    k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wyznacza obcięty rozkład według wartości osobliwych (Truncated SVD / Low-Rank Approximation).

    Zwraca:
        u: macierz bazowa przestrzeni pojęć (m x k),
        s: wektor k największych wartości osobliwych,
        low_rank_docs: reprezentacja dokumentów w zredukowanej przestrzeni (k x n).
    """
    u, s, vt = svds(matrix, k=k)
    # Odwracamy kolejność, aby największe wartości osobliwe były na początku
    u = u[:, ::-1]
    s = s[::-1]
    vt = vt[::-1, :]
    low_rank_docs = np.diag(s) @ vt

    return u, s, low_rank_docs


def vectorize_query(
    query: str,
    word_to_idx: Dict[str, int],
    idf_values: np.ndarray
) -> np.ndarray:
    """
    Przekształca tekst zapytania użytkownika w wektor cech i waży go wartościami IDF.
    """
    q = np.zeros((len(word_to_idx), 1))
    query_tokens = clean_text(query)
    for token in query_tokens:
        if token in word_to_idx:
            idx = word_to_idx[token]
            q[idx, 0] += 1
    return q * idf_values.reshape(-1, 1)


def search_classic(
    query: str,
    word_to_idx: Dict[str, int],
    idf_values: np.ndarray,
    matrix_norm: csr_matrix,
    titles: List[str],
    k: int = 5
) -> List[Tuple[str, float]]:
    """
    Wyszukiwanie klasyczne metodą wektorową (TF-IDF z miarą cosinusową).
    """
    q = vectorize_query(query, word_to_idx, idf_values)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return []

    q_normalized = q / q_norm
    similarities = (q_normalized.T @ matrix_norm).flatten()
    top_indices = np.argsort(similarities)[::-1][:k]

    return [(titles[i], float(similarities[i])) for i in top_indices if similarities[i] > 0]


def search_semantic(
    query: str,
    word_to_idx: Dict[str, int],
    idf_values: np.ndarray,
    u_k: np.ndarray,
    low_rank_docs_norm: np.ndarray,
    titles: List[str],
    k: int = 5
) -> List[Tuple[str, float]]:
    """
    Wyszukiwanie semantyczne (LSA) poprzez rzutowanie zapytania na podprzestrzeń U_k.
    """
    q = vectorize_query(query, word_to_idx, idf_values)
    q_k = u_k.T @ q
    q_k_norm = np.linalg.norm(q_k)
    if q_k_norm == 0:
        return []

    q_k_normalized = q_k / q_k_norm
    similarities = (q_k_normalized.T @ low_rank_docs_norm).flatten()
    top_indices = np.argsort(similarities)[::-1][:k]

    return [(titles[i], float(similarities[i])) for i in top_indices if similarities[i] > 0]
