import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional


class EmbeddingCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                output_dimensionality INTEGER NOT NULL,
                mode TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        with self.lock:
            self.connection.close()

    @staticmethod
    def make_key(model: str, output_dimensionality: int, mode: str, text: str) -> str:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        digest = hashlib.sha256(
            f"{model}|{output_dimensionality}|{mode}|{text_hash}".encode("utf-8")
        ).hexdigest()
        return digest

    def get(self, model: str, output_dimensionality: int, mode: str, text: str) -> Optional[list[float]]:
        key = self.make_key(model, output_dimensionality, mode, text)
        with self.lock:
            row = self.connection.execute(
                "SELECT vector_json FROM embedding_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def put(self, model: str, output_dimensionality: int, mode: str, text: str, vector: list[float]) -> None:
        key = self.make_key(model, output_dimensionality, mode, text)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self.lock:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO embedding_cache
                (key, model, output_dimensionality, mode, text_hash, vector_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (key, model, output_dimensionality, mode, text_hash, json.dumps(vector)),
            )
            self.connection.commit()

    def clear(self) -> None:
        with self.lock:
            self.connection.execute("DELETE FROM embedding_cache")
            self.connection.commit()
