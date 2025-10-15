#!/usr/bin/env python3
import json
import os
import sys

try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)

tool = data.get("tool_name", "")
ti = data.get("tool_input", {})
path = ti.get("file_path") or ti.get("path") or ""

# Normalize the target path to a project-relative form so absolute paths are treated consistently.
proj_root = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
proj_root = os.path.abspath(proj_root)
if path:
    if os.path.isabs(path):
        abs_path = os.path.abspath(path)
    else:
        abs_path = os.path.abspath(os.path.join(proj_root, path))
    try:
        rel_path = os.path.relpath(abs_path, proj_root)
    except ValueError:
        # On Windows, relpath may fail if drives differ; treat as outside project.
        rel_path = path
else:
    rel_path = ""

# Only allow direct writes to specific areas; otherwise require patch packages.
ALLOWED = (
    "patches/",
    "tickets/work_items.json",
    "spec.md",
    "contracts/",
    ".claude/agents/",
)
blocked = True
for a in ALLOWED:
    if rel_path == a.rstrip("/") or rel_path.startswith(a):
        blocked = False
        break

if blocked:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Direct writes to source files are disabled. Please write a patch package to patches/ instead.",
                }
            }
        )
    )
    sys.exit(0)

sys.exit(0)
