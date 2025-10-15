#!/usr/bin/env python3
import json
import re
import sys

try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)

cmd = data.get("tool_input", {}).get("command", "")
danger = [
    r"\brm\s+-rf\s+(/|\.\.)",
    r"curl\s+[^|]+\|\s*sh",
    r"\bsudo\b",
    r"\bdd\s+if=",
    r"\bmkfs\.[a-z0-9]+",
    r"git\s+push\s+.*--force",
]
if any(re.search(p, cmd) for p in danger):
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Blocked potentially dangerous Bash command: {cmd[:80]}...",
                }
            }
        )
    )
    sys.exit(0)

# allow by default
sys.exit(0)
