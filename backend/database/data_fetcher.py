import sqlite3
from typing import List, Tuple


def fetch_articles_from_database(db_path: str, limit: int) -> List[Tuple[str, str]]:
    """
    Pobiera artykuły (tytuł i treść) z lokalnej bazy danych SQLite z limitem rekordów.
    """
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT title, content FROM articles LIMIT ?", (limit,))
        return cursor.fetchall()


def get_article_content(db_path: str, title: str) -> str:
    """
    Zwraca treść pojedynczego artykułu na podstawie jego tytułu.
    """
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT content FROM articles WHERE title = ?", (title,))
        result = cursor.fetchone()
        if result is None or not result[0]:
            return "Brak treści artykułu w bazie danych."
        return result[0]