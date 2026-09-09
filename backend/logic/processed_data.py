import os
from typing import Any, Dict, Optional
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(BASE_DIR, 'data', 'search_engine.joblib')


def save_processed_data(data: Dict[str, Any], path: str = CACHE_PATH) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    print("Zapisywanie przetworzonych danych do pliku")
    joblib.dump(data, path)


def load_processed_data(path: str = CACHE_PATH) -> Optional[Dict[str, Any]]:
    if os.path.exists(path):
        print("Wczytywanie przetworzonych danych do pliku")
        return joblib.load(path)
    return None