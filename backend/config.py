import os
import nltk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data', 'wikipedia.db')

# Domyślny rząd aproksymacji niskorzędowej SVD
K_FACTOR = int(os.getenv('K_FACTOR', 300))

# Maksymalna liczba artykułów przetwarzana do pamięci
MAX_DOCS = int(os.getenv('MAX_DOCS', 300_000))


def init_resources() -> None:
    """Pobiera niezbędne pakiety zasobów lingwistycznych biblioteki NLTK."""
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
