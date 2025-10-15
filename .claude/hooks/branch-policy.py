#!/usr/bin/env python3
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime

PROTECTED_BRANCHES = {"main", "master", "HEAD"}
PROJECT_ROOT = os.path.abspath(os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd()))


def load_payload() -> dict:
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def run_git(args) -> tuple[int, str, str]:
    completed = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def current_branch() -> str | None:
    rc, out, _ = run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return out if rc == 0 else None


def sanitize_branch_name(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    slug = slug[:max_len].rstrip("-")
    return slug or "task"


def ensure_branch(branch_name: str) -> tuple[bool, bool]:
    rc, _, _ = run_git(["git", "rev-parse", "--verify", branch_name])
    if rc == 0:
        rc, _, _ = run_git(["git", "checkout", branch_name])
        return rc == 0, True

    rc, _, _ = run_git(["git", "checkout", "-b", branch_name])
    return rc == 0, False


def feature_branch_from_text(text: str) -> str:
    branch_slug = sanitize_branch_name(" ".join(text.split()[:5]))
    timestamp = datetime.utcnow().strftime("%m%d")
    hash_suffix = hashlib.sha256(text.encode()).hexdigest()[:6]
    return f"feature/{branch_slug}-{timestamp}-{hash_suffix}"


def has_uncommitted_changes() -> bool:
    """Check if there are uncommitted changes in the working tree"""
    rc, out, _ = run_git(["git", "status", "--porcelain"])
    return rc == 0 and bool(out.strip())


def allow(message: str | None = None) -> None:
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
        }
    }
    if message:
        payload["hookSpecificOutput"]["additionalContext"] = message
    print(json.dumps(payload))
    sys.exit(0)


def deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )
    sys.exit(0)


def auto_branch_for_task(payload: dict) -> None:
    # Check for uncommitted changes before attempting branch switch
    if has_uncommitted_changes():
        deny(
            "Cannot auto-create feature branch: uncommitted changes detected. "
            "Please commit or stash your changes first, then retry."
        )

    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        tool_input = {}

    description = tool_input.get("description", "").strip()
    prompt = tool_input.get("prompt", "").strip()

    source_text = description or prompt
    if not source_text:
        fallback_name = datetime.utcnow().strftime("feature/agent-task-%m%d-%H%M%S")
        success, existed = ensure_branch(fallback_name)
        if success:
            state = "switched to" if existed else "created"
            allow(f"Auto-{state} feature branch '{fallback_name}' for agent task.")
        else:
            deny(
                "Failed to auto-create feature branch. "
                "Please run `git checkout -b feature/<name>` and retry."
            )

    candidate = feature_branch_from_text(source_text)
    success, existed = ensure_branch(candidate)
    if success:
        state = "switched to" if existed else "created"
        allow(f"Auto-{state} feature branch '{candidate}' for agent task.")

    fallback_name = datetime.utcnow().strftime("feature/agent-task-%m%d-%H%M%S")
    success, existed = ensure_branch(fallback_name)
    if success:
        state = "switched to" if existed else "created"
        allow(f"Auto-{state} feature branch '{fallback_name}' for agent task.")

    deny(
        "Failed to auto-create feature branch. "
        "Please run `git checkout -b feature/<name>` and retry."
    )


def main() -> None:
    payload = load_payload()
    branch = current_branch()
    if not branch:
        return

    if branch not in PROTECTED_BRANCHES:
        return

    tool_name = payload.get("tool_name", "")
    if tool_name == "Task":
        auto_branch_for_task(payload)
        return

    deny(
        f"Cannot use {tool_name or 'this tool'} on protected branch '{branch}'. "
        "Create a feature branch with `git checkout -b feature/<name>` first."
    )


if __name__ == "__main__":
    main()
