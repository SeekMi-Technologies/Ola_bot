"""SessionManager per-admin path resolution (Ola N2.2)."""

from pathlib import Path

import pytest

from nanobot.agent.admin_context import (
    set_acting_admin_id,
    with_acting_admin_id,
)
from nanobot.session.manager import SessionManager


@pytest.fixture(autouse=True)
def _reset_admin_context():
    set_acting_admin_id(None)
    yield
    set_acting_admin_id(None)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "workspace"


def test_sessions_dir_is_per_admin(workspace: Path) -> None:
    mgr = SessionManager(workspace)
    with with_acting_admin_id("admin-A"):
        path_a = mgr.sessions_dir
    with with_acting_admin_id("admin-B"):
        path_b = mgr.sessions_dir
    assert path_a == workspace / "admins" / "admin-A" / "sessions"
    assert path_b == workspace / "admins" / "admin-B" / "sessions"
    assert path_a != path_b


def test_no_admin_context_falls_back_to_system(workspace: Path) -> None:
    mgr = SessionManager(workspace)
    assert mgr.sessions_dir == workspace / "admins" / "_system" / "sessions"


def test_save_lands_under_acting_admin_dir(workspace: Path) -> None:
    mgr = SessionManager(workspace)
    with with_acting_admin_id("admin-A"):
        session = mgr.get_or_create("api:user:admin-A:conv:xxx")
        session.add_message("user", "hello from A")
        mgr.save(session)
    expected = workspace / "admins" / "admin-A" / "sessions" / "api_user_admin-A_conv_xxx.jsonl"
    assert expected.exists()


def test_two_admins_dont_see_each_others_sessions_on_disk(workspace: Path) -> None:
    mgr = SessionManager(workspace)

    with with_acting_admin_id("admin-A"):
        s_a = mgr.get_or_create("api:user:admin-A:conv:111")
        s_a.add_message("user", "A secret")
        mgr.save(s_a)

    with with_acting_admin_id("admin-B"):
        s_b = mgr.get_or_create("api:user:admin-B:conv:222")
        s_b.add_message("user", "B secret")
        mgr.save(s_b)

    a_files = list((workspace / "admins" / "admin-A" / "sessions").glob("*.jsonl"))
    b_files = list((workspace / "admins" / "admin-B" / "sessions").glob("*.jsonl"))
    assert len(a_files) == 1
    assert len(b_files) == 1
    a_text = a_files[0].read_text(encoding="utf-8")
    b_text = b_files[0].read_text(encoding="utf-8")
    assert "A secret" in a_text and "B secret" not in a_text
    assert "B secret" in b_text and "A secret" not in b_text


def test_list_sessions_returns_only_current_admin(workspace: Path) -> None:
    mgr = SessionManager(workspace)

    with with_acting_admin_id("admin-A"):
        s = mgr.get_or_create("api:user:admin-A:conv:1")
        s.add_message("user", "hi")
        mgr.save(s)

    with with_acting_admin_id("admin-B"):
        s = mgr.get_or_create("api:user:admin-B:conv:2")
        s.add_message("user", "hi")
        mgr.save(s)

    with with_acting_admin_id("admin-A"):
        sessions_a = mgr.list_sessions()
    with with_acting_admin_id("admin-B"):
        sessions_b = mgr.list_sessions()

    assert len(sessions_a) == 1
    assert len(sessions_b) == 1
    assert sessions_a[0]["key"] != sessions_b[0]["key"]


def test_cache_does_not_leak_between_admins_on_same_session_key(workspace: Path) -> None:
    """Channel sessions share keys like 'whatsapp:1234' across admins.
    Cache must not return admin A's Session object to admin B."""
    mgr = SessionManager(workspace)

    with with_acting_admin_id("admin-A"):
        s_a = mgr.get_or_create("whatsapp:1234")
        s_a.add_message("user", "A's message")
        mgr.save(s_a)

    with with_acting_admin_id("admin-B"):
        s_b = mgr.get_or_create("whatsapp:1234")
    assert s_b.messages == []
    assert s_b is not s_a


def test_get_or_create_loads_from_correct_admin_dir(workspace: Path) -> None:
    mgr1 = SessionManager(workspace)
    with with_acting_admin_id("admin-A"):
        s = mgr1.get_or_create("api:user:admin-A:conv:abc")
        s.add_message("user", "persisted")
        mgr1.save(s)

    mgr2 = SessionManager(workspace)
    with with_acting_admin_id("admin-A"):
        loaded = mgr2.get_or_create("api:user:admin-A:conv:abc")
    assert len(loaded.messages) == 1
    assert loaded.messages[0]["content"] == "persisted"

    mgr3 = SessionManager(workspace)
    with with_acting_admin_id("admin-B"):
        not_loaded = mgr3.get_or_create("api:user:admin-A:conv:abc")
    assert not_loaded.messages == []


def test_delete_session_only_removes_acting_admin_file(workspace: Path) -> None:
    mgr = SessionManager(workspace)
    with with_acting_admin_id("admin-A"):
        s = mgr.get_or_create("shared-key")
        s.add_message("user", "A")
        mgr.save(s)
    with with_acting_admin_id("admin-B"):
        s = mgr.get_or_create("shared-key")
        s.add_message("user", "B")
        mgr.save(s)

    with with_acting_admin_id("admin-A"):
        removed = mgr.delete_session("shared-key")
    assert removed
    assert not (workspace / "admins" / "admin-A" / "sessions" / "shared-key.jsonl").exists()
    assert (workspace / "admins" / "admin-B" / "sessions" / "shared-key.jsonl").exists()


def test_flush_all_writes_each_session_under_its_own_admin(workspace: Path) -> None:
    mgr = SessionManager(workspace)
    with with_acting_admin_id("admin-A"):
        s = mgr.get_or_create("api:user:admin-A:conv:1")
        s.add_message("user", "A")
    with with_acting_admin_id("admin-B"):
        s = mgr.get_or_create("api:user:admin-B:conv:2")
        s.add_message("user", "B")

    set_acting_admin_id(None)
    flushed = mgr.flush_all()
    assert flushed == 2

    a_files = list((workspace / "admins" / "admin-A" / "sessions").glob("*.jsonl"))
    b_files = list((workspace / "admins" / "admin-B" / "sessions").glob("*.jsonl"))
    assert len(a_files) == 1
    assert len(b_files) == 1
    assert "A" in a_files[0].read_text(encoding="utf-8")
    assert "B" in b_files[0].read_text(encoding="utf-8")
