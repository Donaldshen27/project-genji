#!/usr/bin/env python3
import json, sys, os, re

try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)

ti = data.get("tool_input",{})
path = ti.get("path") or ti.get("file_path") or ti.get("uri") or ""

def risky(p):
    bad = [".env", ".git/", "/.git/", "/etc/", "/var/"]
    return any(b in p for b in bad)

allowed_roots = ("src/", "tests/", "contracts/", "tickets/", "patches/")
decision = "ask"
reason = "Requesting confirmation for MCP write."

if path:
    if risky(path):
        decision = "deny"; reason = f"Forbidden path: {path}"
    elif not path.startswith(allowed_roots) and not path.startswith("./"+allowed_roots[0]):
        decision = "ask"; reason = f"Attempting to write outside the allowed roots: {path}. Proceed?"

print(json.dumps({
    "hookSpecificOutput": {
      "hookEventName": "PreToolUse",
      "permissionDecision": decision,
      "permissionDecisionReason": reason
    }
}))
