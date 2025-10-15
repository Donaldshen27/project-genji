#!/usr/bin/env python3
import json
import sys

try:
    data = json.load(sys.stdin)
except Exception:
    data = {}
print(
    json.dumps(
        {
            "continue": True,
            "systemMessage": "PreCompact: Keep artifacts concise; prefer ledger summaries over full logs.",
        }
    )
)
