#!/usr/bin/env python3

"""Archive completed work items from tickets/work_items.json.

Usage examples:
    python scripts/archive_work_items.py --work-item SMK-001-001
    python scripts/archive_work_items.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ITEMS_PATH = REPO_ROOT / "tickets" / "work_items.json"
ARCHIVE_DIR = REPO_ROOT / "tickets" / "archive"


def load_work_items() -> dict:
    if not WORK_ITEMS_PATH.exists():
        raise FileNotFoundError(f"Missing {WORK_ITEMS_PATH}")
    with WORK_ITEMS_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_work_items(payload: dict) -> None:
    with WORK_ITEMS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def ensure_archive_dir() -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_slug(raw: str | None) -> str:
    if not raw:
        return "ticket"
    slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in raw)
    slug = slug.strip("-_")
    return slug or "ticket"


def archive_modules(
    ticket_payload: dict, archived: Iterable[dict], archive_file: Path
) -> None:
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    with archive_file.open("a", encoding="utf-8") as fh:
        for module in archived:
            fh.write(
                json.dumps(
                    {
                        "archived_at": now,
                        "ticket_key": ticket_payload.get("ticket_key"),
                        "work_item_id": module.get("work_item_id"),
                        "module": module,
                    }
                )
            )
            fh.write("\n")


def recompute_metadata(payload: dict, remaining_modules: list[dict]) -> None:
    payload["modules"] = remaining_modules
    payload["total_work_items"] = len(remaining_modules)

    total_tokens = 0
    for module in remaining_modules:
        tokens = module.get("estimated_tokens") or module.get("estimated_total_tokens")
        if isinstance(tokens, int):
            total_tokens += tokens
    payload["estimated_total_tokens"] = total_tokens

    remaining_ids = {
        module.get("work_item_id") for module in remaining_modules if module.get("work_item_id")
    }

    if isinstance(payload.get("critical_path"), list):
        payload["critical_path"] = [
            wid for wid in payload["critical_path"] if wid in remaining_ids
        ]

    if isinstance(payload.get("parallel_executable"), list):
        payload["parallel_executable"] = [
            wid for wid in payload["parallel_executable"] if wid in remaining_ids
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive work items from tickets/work_items.json"
    )
    parser.add_argument(
        "--work-item",
        "-w",
        dest="work_items",
        action="append",
        help="Work item ID to archive (can be repeated)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Archive all work items present in tickets/work_items.json",
    )
    args = parser.parse_args()

    if not args.all and not args.work_items:
        parser.error("Provide --work-item/ -w at least once or use --all.")

    return args


def main() -> int:
    args = parse_args()

    try:
        payload = load_work_items()
    except Exception as exc:
        print(f"error: failed to read work items: {exc}", file=sys.stderr)
        return 1

    modules = payload.get("modules")
    if not isinstance(modules, list):
        print("error: work_items.json missing 'modules' array", file=sys.stderr)
        return 1

    if args.all:
        target_ids = {module.get("work_item_id") for module in modules if module.get("work_item_id")}
    else:
        target_ids = set(args.work_items)

    if not target_ids:
        print("error: no valid work item IDs to archive", file=sys.stderr)
        return 1

    modules_by_id = {
        module.get("work_item_id"): module for module in modules if module.get("work_item_id")
    }

    missing_ids = [wid for wid in target_ids if wid not in modules_by_id]
    if missing_ids:
        print(f"error: unknown work item IDs: {', '.join(missing_ids)}", file=sys.stderr)
        return 1

    archived_modules = [modules_by_id[wid] for wid in target_ids]
    remaining_modules = [m for m in modules if m.get("work_item_id") not in target_ids]

    ensure_archive_dir()
    ticket_slug = sanitize_slug(payload.get("ticket_key") or payload.get("goal"))
    archive_path = ARCHIVE_DIR / f"{ticket_slug}.jsonl"
    archive_modules(payload, archived_modules, archive_path)

    recompute_metadata(payload, remaining_modules)

    try:
        save_work_items(payload)
    except Exception as exc:
        print(f"error: failed to write updated work_items.json: {exc}", file=sys.stderr)
        return 1

    archived_ids = ", ".join(sorted(target_ids))
    print(
        f"Archived work items: {archived_ids}\n"
        f"- archive file: {archive_path}\n"
        f"- remaining work items: {len(remaining_modules)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
