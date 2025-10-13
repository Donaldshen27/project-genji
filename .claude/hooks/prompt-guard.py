#!/usr/bin/env python3
import json, sys, re
try:
    data = json.load(sys.stdin)
except Exception as e:
    sys.exit(0)

prompt = data.get("prompt","")
# Very simple secret detection and 'Ticket:' hint injector
if re.search(r"(?i)(password|secret|api[-_ ]?key|token)\s*[:=]", prompt):
    print(json.dumps({
        "decision": "block",
        "reason": "Security policy: The prompt appears to contain a secret. Remove or obfuscate it, then retry."
    }))
    sys.exit(0)

# Encourage ticket header for implementer/skeletoner flows
if "implementer" in prompt.lower() and "ticket:" not in prompt.lower():
    extra = "Reminder: Include a header line 'Ticket: KEY-123' so the hooks can detect the scope."
    print(json.dumps({
      "hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": extra
      }
    }))
    sys.exit(0)

# default: allow
sys.exit(0)
