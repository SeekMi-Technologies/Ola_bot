#!/usr/bin/env python3
"""One-shot migration for Ola N2 per-admin workspace layout.

Before (legacy, all admins share):
  workspace/
    sessions/api_user_<adminId>_conv_*.jsonl
    memory/MEMORY.md
    memory/history.jsonl
    USER.md
    SOUL.md / AGENTS.md / TOOLS.md / HEARTBEAT.md / skills/  (global)

After (per-admin isolated):
  workspace/
    SOUL.md / AGENTS.md / TOOLS.md / HEARTBEAT.md / skills/  (unchanged)
    admins/
      <adminId>/
        sessions/
        memory/MEMORY.md
        memory/history.jsonl
        USER.md

Per Ola zyd's decision: wipe legacy session / memory / USER.md data
rather than heuristic-migrate. Existing data is mostly testing artifacts
and the cross-admin contamination in MEMORY.md / history.jsonl /
USER.md cannot be reliably split apart by adminId after the fact.

Usage:
    python scripts/migrate_workspace_to_per_admin.py --workspace ~/.nanobot/workspace
    python scripts/migrate_workspace_to_per_admin.py --workspace ~/.nanobot/workspace --yes  # skip confirmation
    python scripts/migrate_workspace_to_per_admin.py --workspace ~/.nanobot/workspace --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path


WIPE_TARGETS = ["sessions", "memory", "USER.md"]
PRESERVE = {"SOUL.md", "AGENTS.md", "TOOLS.md", "HEARTBEAT.md", "skills", ".git", ".gitignore", "admins"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate nanobot workspace to per-admin layout (Ola N2).")
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would change without touching files.")
    parser.add_argument("--no-backup", action="store_true", help="Skip the zip backup (NOT recommended).")
    args = parser.parse_args()

    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        print(f"Error: workspace {workspace} is not a directory.", file=sys.stderr)
        return 1

    targets = [workspace / name for name in WIPE_TARGETS]
    existing = [p for p in targets if p.exists()]

    print(f"Workspace: {workspace}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()
    if not existing:
        print("Nothing to wipe — workspace already has no legacy session/memory/USER data.")
        return 0

    print("Will wipe (legacy global, cross-admin contaminated):")
    for p in existing:
        kind = "dir" if p.is_dir() else "file"
        print(f"  - {p} ({kind})")
    print()
    print("Will preserve (global / per-tenant):")
    for name in sorted(PRESERVE):
        p = workspace / name
        if p.exists():
            print(f"  - {p}")
    print()

    if args.dry_run:
        print("[dry-run] No changes made. Re-run without --dry-run to apply.")
        return 0

    if not args.yes:
        reply = input("Proceed? [y/N] ").strip().lower()
        if reply != "y":
            print("Aborted.")
            return 1

    if not args.no_backup:
        backup = workspace.parent / f"{workspace.name}.backup.{int(time.time())}"
        print(f"Creating backup at {backup}.zip ...")
        shutil.make_archive(str(backup), "zip", workspace)
        print(f"Backup complete: {backup}.zip")
        print()

    for p in existing:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        print(f"Wiped: {p}")

    admins_dir = workspace / "admins"
    admins_dir.mkdir(exist_ok=True)
    print()
    print(f"Created: {admins_dir}/")
    print()
    print("Migration complete. Restart nanobot serve.")
    print("New session/memory/USER.md files will land under workspace/admins/<adminId>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
