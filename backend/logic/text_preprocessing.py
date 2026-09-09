from typing import List, Optional
import nltk
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
try:
    STOPWORDS = set(nltk.corpus.stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(nltk.corpus.stopwords.words('english'))


def clean_text(text: Optional[str]) -> List[str]:
    """
    Oczyszcza i normalizuje tekst:
    - zamienia na małe litery,
    - tokenizuje,
    - usuwa znaki niebędące alfanumerycznymi oraz stopwords,
    - dokonuje stemmingu za pomocą algorytmu Portera.
    """
    if not text:
        return []

    text_lower = text.lower()
    tokens = nltk.word_tokenize(text_lower)
    filtered = [token for token in tokens if token.isalnum() and token not in STOPWORDS]
    stemmed = [stemmer.stem(token) for token in filtered]
    return stemmed