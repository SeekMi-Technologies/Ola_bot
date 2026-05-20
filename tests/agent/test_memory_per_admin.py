"""MemoryStore per-admin file paths (Ola N2.3)."""

from pathlib import Path

import pytest

from nanobot.agent.admin_context import (
    set_acting_admin_id,
    with_acting_admin_id,
)
from nanobot.agent.memory import MemoryStore


@pytest.fixture(autouse=True)
def _reset_admin_context():
    set_acting_admin_id(None)
    yield
    set_acting_admin_id(None)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path)


def test_memory_paths_are_per_admin(store, tmp_path):
    with with_acting_admin_id("admin-A"):
        a_memory = store.memory_file
        a_history = store.history_file
        a_user = store.user_file
        a_cursor = store._cursor_file
    with with_acting_admin_id("admin-B"):
        b_memory = store.memory_file
    assert a_memory == tmp_path / "admins" / "admin-A" / "memory" / "MEMORY.md"
    assert a_history == tmp_path / "admins" / "admin-A" / "memory" / "history.jsonl"
    assert a_user == tmp_path / "admins" / "admin-A" / "USER.md"
    assert a_cursor == tmp_path / "admins" / "admin-A" / "memory" / ".cursor"
    assert b_memory == tmp_path / "admins" / "admin-B" / "memory" / "MEMORY.md"
    assert a_memory != b_memory


def test_soul_stays_global(store, tmp_path):
    with with_acting_admin_id("admin-A"):
        a_soul = store.soul_file
    with with_acting_admin_id("admin-B"):
        b_soul = store.soul_file
    assert a_soul == tmp_path / "SOUL.md"
    assert b_soul == tmp_path / "SOUL.md"


def test_no_admin_context_uses_system_dir(store, tmp_path):
    assert store.memory_file == tmp_path / "admins" / "_system" / "memory" / "MEMORY.md"
    assert store.user_file == tmp_path / "admins" / "_system" / "USER.md"


def test_write_memory_isolated_between_admins(store):
    with with_acting_admin_id("admin-A"):
        store.write_memory("admin A long-term facts")
    with with_acting_admin_id("admin-B"):
        store.write_memory("admin B long-term facts")
        assert store.read_memory() == "admin B long-term facts"
    with with_acting_admin_id("admin-A"):
        assert store.read_memory() == "admin A long-term facts"


def test_write_user_isolated_between_admins(store):
    with with_acting_admin_id("admin-A"):
        store.write_user("# A's profile")
    with with_acting_admin_id("admin-B"):
        store.write_user("# B's profile")
        assert store.read_user() == "# B's profile"
    with with_acting_admin_id("admin-A"):
        assert store.read_user() == "# A's profile"


def test_append_history_isolated_between_admins(store):
    with with_acting_admin_id("admin-A"):
        store.append_history("A entry 1")
        store.append_history("A entry 2")
    with with_acting_admin_id("admin-B"):
        store.append_history("B entry 1")

    with with_acting_admin_id("admin-A"):
        a_entries = store.read_unprocessed_history(since_cursor=0)
    with with_acting_admin_id("admin-B"):
        b_entries = store.read_unprocessed_history(since_cursor=0)

    assert len(a_entries) == 2
    assert "A entry 1" in a_entries[0]["content"]
    assert "A entry 2" in a_entries[1]["content"]
    assert len(b_entries) == 1
    assert "B entry 1" in b_entries[0]["content"]


def test_gitstore_tracks_only_global_soul(store):
    assert store.git._tracked_files == ["SOUL.md"]


def test_read_memory_empty_for_fresh_admin(store):
    with with_acting_admin_id("admin-new"):
        assert store.read_memory() == ""
        assert store.read_user() == ""
        assert store.read_unprocessed_history(since_cursor=0) == []
