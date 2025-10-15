#!/usr/bin/env python3
import json
import os
import re
import sys


def approx_tokens(s: str) -> int:
    # crude: ~4 chars per token
    return max(1, len(s) // 4)


proj = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
budgets = {"planner": 6000, "skeletoner": 4000, "implementer": 2500, "integrator": 2000}
by_role_path = os.path.join(proj, "orchestrator", "budgets.yaml")
if os.path.exists(by_role_path):
    # best-effort parse simple yaml "key: {max_tokens: N}"
    try:
        import re

        content = open(by_role_path, encoding="utf-8").read()
        for line in content.splitlines():
            m = re.match(r"(\w+):\s*\{max_tokens:\s*(\d+)\s*\}", line.strip())
            if m:
                budgets[m.group(1)] = int(m.group(2))
    except Exception:
        pass

try:
    hook_in = json.load(sys.stdin)
except Exception:
    hook_in = {}

tool_name = hook_in.get("tool_name", "")
role_guess = "implementer" if "Task" in tool_name or True else "implementer"
pack = {}
try:
    pack = json.load(open("/tmp/context.json", encoding="utf-8"))
except Exception:
    pass

blob = json.dumps(pack, ensure_ascii=False)
tok = approx_tokens(blob)
limit = budgets.get(role_guess, 2500)

if tok > limit:
    # Block continuation with a clear message to the model/user
    out = {
        "continue": False,
        "stopReason": f"Context pack (~{tok} tokens) exceeds the role budget ({limit}).",
        "systemMessage": "Shrink the scope: reduce the file_region window, drop unrelated neighbors, or split the ticket.",
    }
    print(json.dumps(out))
    sys.exit(0)
sys.exit(0)
