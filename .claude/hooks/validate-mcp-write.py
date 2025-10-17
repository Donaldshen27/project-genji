#!/usr/bin/env python3
import json
import os
import sys

try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)

ti = data.get("tool_input", {})
raw_path = ti.get("path") or ti.get("file_path") or ti.get("uri") or ""


def risky(p: str) -> bool:
    bad = [".env", ".git/", "/.git/", "/etc/", "/var/"]
    return any(b in p for b in bad)


allowed_roots = ("src/", "tests/", "contracts/", "tickets/", "patches/")
decision = "ask"
reason = "Requesting confirmation for MCP write."

proj_root = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
proj_root = os.path.abspath(proj_root)
normalized = raw_path

if raw_path:
    if os.path.isabs(raw_path):
        abs_path = os.path.abspath(raw_path)
    else:
        abs_path = os.path.abspath(os.path.join(proj_root, raw_path))
    try:
        rel_path = os.path.relpath(abs_path, proj_root)
    except ValueError:
        rel_path = raw_path
    normalized = rel_path

    if risky(raw_path) or risky(rel_path):
        decision = "deny"
        reason = f"Forbidden path: {raw_path}"
    else:
        for prefix in allowed_roots:
            clean_prefix = prefix.rstrip("/")
            if normalized == clean_prefix or normalized.startswith(prefix):
                decision = "allow"
                reason = "Auto-approved write within allowed roots."
                break
        else:
            decision = "ask"
            reason = f"Attempting to write outside the allowed roots: {raw_path}. Proceed?"

print(
    json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": decision,
                "permissionDecisionReason": reason,
            }
        }
    )
)
