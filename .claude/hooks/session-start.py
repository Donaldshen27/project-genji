#!/usr/bin/env python3
import json

# Add a short operating blurb to context on session start
out = {
    "hookSpecificOutput": {
        "hookEventName": "SessionStart",
        "additionalContext": (
            "Operating mode: Ticketized patches with deterministic hooks. "
            "Work from a feature branch (not main) named after your ticket. "
            "Implementers must write JSON patch packages into patches/. "
            "The PostToolUse pipeline will apply, format, test, and commit."
        ),
    }
}
print(json.dumps(out))
