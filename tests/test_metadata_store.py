import time

import pytest

from semcache.stores.metadata_store import MetadataStore


@pytest.fixture
def store(tmp_path):
    return MetadataStore(db_path=str(tmp_path / "metadata.sqlite"))


class TestMetadataStore:
    def test_insert_entry(self, store):
        store.add_entry(1, "Hello?", "hello", "Hi there")
        assert store.get_total_entries() == 1

    def test_retrieve_entry(self, store):
        store.add_entry(7, "What is AI?", "what is ai", "AI is...")
        row = store.get_entry(7)
        assert row is not None
        assert row["vector_id"] == 7
        assert row["prompt"] == "What is AI?"
        assert row["normalized_prompt"] == "what is ai"
        assert row["response"] == "AI is..."
        assert row["hit_count"] == 0

    def test_get_entry_missing_returns_none(self, store):
        assert store.get_entry(999) is None

    def test_update_access_time_increments_hit_count(self, store):
        store.add_entry(1, "q", "q", "a")
        store.update_access_time(1)
        row = store.get_entry(1)
        assert row["hit_count"] == 1

    def test_update_access_time_refreshes_timestamp(self, store):
        store.add_entry(1, "q", "q", "a")
        before = store.get_entry(1)["last_accessed"]
        time.sleep(0.01)
        store.update_access_time(1)
        after = store.get_entry(1)["last_accessed"]
        assert after > before

    def test_multiple_updates_accumulate_hit_count(self, store):
        store.add_entry(1, "q", "q", "a")
        store.update_access_time(1)
        store.update_access_time(1)
        store.update_access_time(1)
        assert store.get_entry(1)["hit_count"] == 3

    def test_delete_entry(self, store):
        store.add_entry(1, "q", "q", "a")
        store.delete_entry(1)
        assert store.get_entry(1) is None
        assert store.get_total_entries() == 0

    def test_delete_entry_removes_from_prompt_index(self, store):
        store.add_entry(1, "q", "normalized_q", "a")
        store.delete_entry(1)
        assert "normalized_q" not in store.prompt_index

    def test_delete_nonexistent_is_noop(self, store):
        store.delete_entry(999)  # should not raise

    def test_lru_ordering(self, store):
        store.add_entry(1, "a", "a", "ra")
        time.sleep(0.01)
        store.add_entry(2, "b", "b", "rb")
        time.sleep(0.01)
        store.add_entry(3, "c", "c", "rc")
        rows = store.get_lru_entries()
        ids = [r["vector_id"] for r in rows]
        assert ids == [1, 2, 3]

    def test_lru_ordering_after_access(self, store):
        store.add_entry(1, "a", "a", "ra")
        time.sleep(0.01)
        store.add_entry(2, "b", "b", "rb")
        time.sleep(0.01)
        store.update_access_time(1)
        rows = store.get_lru_entries()
        # entry 2 now oldest (least recently accessed)
        assert rows[0]["vector_id"] == 2

    def test_prompt_index_populated_on_add(self, store):
        store.add_entry(42, "Hello world", "hello world", "response")
        assert store.prompt_index["hello world"] == 42

    def test_prompt_index_loaded_on_startup(self, tmp_path):
        db = str(tmp_path / "metadata.sqlite")
        s1 = MetadataStore(db_path=db)
        s1.add_entry(5, "q", "norm_q", "a")
        # new connection — index must be rebuilt from DB
        s2 = MetadataStore(db_path=db)
        assert s2.prompt_index["norm_q"] == 5

    def test_get_total_entries(self, store):
        assert store.get_total_entries() == 0
        store.add_entry(1, "a", "a", "ra")
        store.add_entry(2, "b", "b", "rb")
        assert store.get_total_entries() == 2
