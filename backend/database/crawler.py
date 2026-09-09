import asyncio
import os
import sqlite3
import sys
from typing import Optional, Tuple
import wikipediaapi

# Zapewnienie dostępu do modułów z katalogu backend
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from config import DB_PATH
except ImportError:
    DB_PATH = os.path.join(BACKEND_DIR, 'data', 'wikipedia.db')

COUNT_OF_ARTICLES = 300_000


def setup_database(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    with connection:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT UNIQUE,
                content TEXT
            )
        ''')
    return connection


async def get_article(wiki: wikipediaapi.AsyncWikipedia, title: str, semaphore: asyncio.Semaphore) -> Tuple[str, Optional[str]]:
    async with semaphore:
        try:
            page = wiki.page(title)
            text = await page.text
            return title, text
        except Exception as e:
            print("Błąd dla: ", title, " e: ", e)
            return title, None


async def main() -> None:
    wiki = wikipediaapi.AsyncWikipedia(
        user_agent='michalswiatek2005@gmail.com',
        language='en',
        max_retries=3,
    )
    connection = setup_database(DB_PATH)
    cursor = connection.cursor()
    semaphore = asyncio.Semaphore(15)

    cursor.execute('SELECT COUNT(*) FROM articles')
    fetched = cursor.fetchone()[0]
    print(f"W bazie jest {fetched} artykulów")

    while fetched < COUNT_OF_ARTICLES:
        try:
            titles = await wiki.random(limit=100)
            tasks = [get_article(wiki, t, semaphore) for t in titles]
            results = await asyncio.gather(*tasks)

            added = 0
            for title, text in results:
                if text and len(text) > 500:
                    try:
                        cursor.execute(
                            "INSERT OR IGNORE INTO articles (title, content) VALUES (?, ?)",
                            (title, text)
                        )
                        if cursor.rowcount > 0:
                            fetched += 1
                            added += 1
                    except Exception:
                        pass

            connection.commit()
            print(f"Pobrano {fetched}/{COUNT_OF_ARTICLES}. Teraz pobrabo: {added}")
            await asyncio.sleep(1)
        except Exception as e:
            print("Błąd głownej pętli: ", e)
            await asyncio.sleep(5)

    connection.close()
    print("Pobieranie zakończone")


if __name__ == '__main__':
    asyncio.run(main())
