import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MetadataStore:
    """SQLite-backed metadata store keyed by FAISS vector ID."""

    DEFAULT_DB = Path.home() / ".semcache" / "metadata.sqlite"

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = Path(db_path) if db_path else self.DEFAULT_DB
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # normalized_prompt → vector_id for O(1) exact lookup
        self.prompt_index: dict[str, int] = {}
        self.initialize_db()
        self._load_prompt_index()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def initialize_db(self) -> None:
        """Create the cache_entries table if it does not exist."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                vector_id        INTEGER PRIMARY KEY,
                prompt           TEXT,
                normalized_prompt TEXT,
                response         TEXT,
                created_at       TIMESTAMP,
                last_accessed    TIMESTAMP,
                hit_count        INTEGER
            )
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_entry(
        self,
        vector_id: int,
        prompt: str,
        normalized_prompt: str,
        response: str,
    ) -> None:
        """Insert a new cache entry."""
        now = _now()
        self._conn.execute(
            """
            INSERT INTO cache_entries
                (vector_id, prompt, normalized_prompt, response,
                 created_at, last_accessed, hit_count)
            VALUES (?, ?, ?, ?, ?, ?, 0)
            """,
            (vector_id, prompt, normalized_prompt, response, now, now),
        )
        self._conn.commit()
        self.prompt_index[normalized_prompt] = vector_id

    def update_access_time(self, vector_id: int) -> None:
        """Increment hit_count and refresh last_accessed."""
        self._conn.execute(
            """
            UPDATE cache_entries
            SET last_accessed = ?, hit_count = hit_count + 1
            WHERE vector_id = ?
            """,
            (_now(), vector_id),
        )
        self._conn.commit()

    def delete_entry(self, vector_id: int) -> None:
        """Remove an entry (used during FAISS eviction)."""
        row = self.get_entry(vector_id)
        if row:
            self.prompt_index.pop(row["normalized_prompt"], None)
        self._conn.execute(
            "DELETE FROM cache_entries WHERE vector_id = ?", (vector_id,)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_entry(self, vector_id: int) -> Optional[dict]:
        """Return entry as a dict, or None if not found."""
        cur = self._conn.execute(
            "SELECT * FROM cache_entries WHERE vector_id = ?", (vector_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_lru_entries(self) -> list[dict]:
        """Return all entries ordered by last_accessed ascending (LRU first)."""
        cur = self._conn.execute(
            "SELECT * FROM cache_entries ORDER BY last_accessed ASC"
        )
        return [dict(r) for r in cur.fetchall()]

    def get_total_entries(self) -> int:
        """Return total number of stored entries."""
        cur = self._conn.execute("SELECT COUNT(*) FROM cache_entries")
        return cur.fetchone()[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_prompt_index(self) -> None:
        """Populate prompt_index from existing rows on startup."""
        cur = self._conn.execute(
            "SELECT vector_id, normalized_prompt FROM cache_entries"
        )
        for row in cur.fetchall():
            self.prompt_index[row["normalized_prompt"]] = row["vector_id"]
