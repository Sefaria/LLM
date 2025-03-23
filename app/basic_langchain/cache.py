import sqlite3
import pickle
from typing import List
from functools import wraps


def ensure_cache_tables_exists():
    _ensure_table_exists('''CREATE TABLE IF NOT EXISTS chat_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model TEXT NOT NULL,
                        temperature REAL NOT NULL,
                        max_tokens INTEGER NOT NULL,
                        messages BLOB NOT NULL,
                        response BLOB NOT NULL
                      )''', '''CREATE INDEX IF NOT EXISTS idx_cache_model_messages_temperature 
                      ON chat_cache (model, messages, temperature, max_tokens)''')
    _ensure_table_exists('''CREATE TABLE IF NOT EXISTS embedding_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model TEXT NOT NULL,
                        messages BLOB NOT NULL,
                        response BLOB NOT NULL
                      )''', '''CREATE INDEX IF NOT EXISTS idx_cache_model_messages
                      ON embedding_cache (model, messages)''')


def _ensure_table_exists(table_def, index_def):
    connection = sqlite3.connect('.llm_cache.db')
    cursor = connection.cursor()
    cursor.execute(table_def)
    cursor.execute(index_def)
    connection.commit()
    connection.close()


ensure_cache_tables_exists()


def _get_query__values(cache_type, instance, messages):
    if cache_type == 'chat':
        query = "SELECT response FROM chat_cache WHERE model = ? AND messages = ? AND temperature = ? AND max_tokens = ?"
        values = (instance.model, pickle.dumps(messages), instance.temperature, getattr(instance, 'max_tokens', 0))
        insert_query = "INSERT INTO chat_cache (model, messages, temperature, max_tokens, response) VALUES (?, ?, ?, ?, ?)"
    elif cache_type == 'embedding':
        query = "SELECT response FROM embedding_cache WHERE model = ? AND messages = ?"
        values = (instance.model, pickle.dumps(messages))
        insert_query = "INSERT INTO embedding_cache (model, messages, response) VALUES (?, ?, ?)"
    else:
        raise Exception("Invalid cache type")
    return query, values, insert_query



# def sqlite_cache(cache_type):
#     """
#     :param cache_type: valid types are 'chat', 'embedding'
#     """
#     def decorator_sqlite_cache(func):
#         @wraps(func)
#         def wrapper(self, messages: list):
#             connection = sqlite3.connect('.llm_cache.db')
#             cursor = connection.cursor()
#             query, values, insert_query = _get_query__values(cache_type, self, messages)
#             cursor.execute(query, values)
#             cached_response = cursor.fetchone()
#
#             if cached_response and False:
#                 response = pickle.loads(cached_response[0])
#             else:
#                 response = func(self, messages)
#                 insert_values = tuple(list(values) + [pickle.dumps(response)])
#                 cursor.execute(insert_query, insert_values)
#                 connection.commit()
#
#             connection.close()
#             return response
#         return wrapper
#     return decorator_sqlite_cache
def sqlite_cache(cache_type):
    """
    :param cache_type: valid types are 'chat', 'embedding'
    """
    def decorator_sqlite_cache(func):
        @wraps(func)
        def wrapper(self, messages: list):
            # Create a new connection for each thread
            connection = sqlite3.connect('.llm_cache.db', check_same_thread=False)
            cursor = connection.cursor()
            query, values, insert_query = _get_query__values(cache_type, self, messages)
            cursor.execute(query, values)
            cached_response = cursor.fetchone()

            if cached_response:
                response = pickle.loads(cached_response[0])
            else:
                response = func(self, messages)
                insert_values = tuple(list(values) + [pickle.dumps(response)])
                cursor.execute(insert_query, insert_values)
                connection.commit()

            connection.close()
            return response
        return wrapper
    return decorator_sqlite_cache