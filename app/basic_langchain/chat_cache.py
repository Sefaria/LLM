import sqlite3
import pickle
from typing import List
from functools import wraps


def ensure_cache_table_exists():
    connection = sqlite3.connect('.llm_cache.db')
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model TEXT NOT NULL,
                        temperature REAL NOT NULL,
                        max_tokens INTEGER NOT NULL,
                        messages BLOB NOT NULL,
                        response BLOB NOT NULL
                      )''')
    cursor.execute('''CREATE INDEX IF NOT EXISTS idx_cache_model_messages_temperature 
                      ON cache (model, messages, temperature, max_tokens)''')
    connection.commit()
    connection.close()


ensure_cache_table_exists()


def sqlite_cache(func):
    @wraps(func)
    def wrapper(self, messages: list):
        connection = sqlite3.connect('.llm_cache.db')
        cursor = connection.cursor()

        query = "SELECT response FROM cache WHERE model = ? AND messages = ? AND temperature = ? AND max_tokens = ?"
        values = (self.model, pickle.dumps(messages), self.temperature, getattr(self, 'max_tokens', 0))

        cursor.execute(query, values)
        cached_response = cursor.fetchone()

        if cached_response:
            response = pickle.loads(cached_response[0])
        else:
            response = func(self, messages)
            insert_query = "INSERT INTO cache (model, messages, temperature, max_tokens, response) VALUES (?, ?, ?, ?, ?)"
            insert_values = tuple(list(values) + [pickle.dumps(response)])
            cursor.execute(insert_query, insert_values)
            connection.commit()

        connection.close()
        return response

    return wrapper