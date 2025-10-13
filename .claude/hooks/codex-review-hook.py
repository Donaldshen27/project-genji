#!/usr/bin/env python3
"""
Codex Review Hook for Claude Code
This script reviews code changes before they're applied using codex.
"""

import datetime
import json
import os
import re
import subprocess
import sys


def review_prompt_for_edit(tool_input: dict) -> str:
    file_path = tool_input.get("file_path", "unknown")
    old_string = tool_input.get("old_string", "")
    new_string = tool_input.get("new_string", "")
    return f"""Review this code change before it's applied:

File: {file_path}

Problem/Context: Code modification requested by user

Proposed Solution:
OLD CODE:
```
{old_string}
```

NEW CODE:
```
{new_string}
```

Please assess this change and respond with one of the following:

1. If the change is correct, safe, and ready to apply, start your response with 'APPROVE:' followed by a brief confirmation.

2. If the change has issues, start your response with 'REJECT:' followed by:
   - What's wrong with the proposed change
   - Specific suggestions for how to fix it
   - What the corrected version should look like

Be critical and thorough. Only approve if you're 100% satisfied."""


def review_prompt_for_write(tool_input: dict) -> str:
    file_path = tool_input.get("file_path", "unknown")
    content = tool_input.get("content", "")
    return f"""Review this new file before it's created:

File: {file_path}

Problem/Context: New file creation requested by user

Proposed Solution:
```
{content}
```

Please assess this new file and respond with one of the following:

1. If the code is correct, safe, and ready to create, start your response with 'APPROVE:' followed by a brief confirmation.

2. If the code has issues, start your response with 'REJECT:' followed by:
   - What's wrong with the proposed code
   - Specific suggestions for how to fix it
   - What the corrected version should look like

Be critical and thorough. Only approve if you're 100% satisfied."""


def review_prompt_for_patch_package(tool_input: dict) -> str:
    ticket_id = tool_input.get("ticket", "unknown ticket")
    diff = tool_input.get("diff", "")
    metadata = tool_input.get("metadata")
    context_block = ""
    if isinstance(metadata, dict) and metadata:
        context_lines = []
        for key, value in metadata.items():
            context_lines.append(f"- {key}: {value}")
        context_txt = "\n".join(context_lines)
        context_block = f"\nContext provided:\n{context_txt}\n"
    return f"""Review this unified diff before it's applied:

Ticket: {ticket_id}

{context_block}
Proposed change (unified diff):
```diff
{diff}
```

Please assess this entire patch set and respond with one of the following:

1. If every change is correct, safe, and ready to merge, start your response with 'APPROVE:' followed by a brief confirmation (feel free to mention any key points you checked).

2. If you spot any issue—incorrect logic, missing context, regressions, tests needed, etc.—start your response with 'REJECT:' followed by:
   - What is wrong
   - Specific suggestions for how to fix it
   - What the corrected code should look like or what additional work is required

Evaluate the diff holistically; approving partial fixes is not allowed. Only approve if you're completely satisfied with the combined changes."""


def build_prompt(tool_name: str, tool_input: dict) -> str | None:
    if tool_name == "Edit":
        return review_prompt_for_edit(tool_input)
    if tool_name == "Write":
        return review_prompt_for_write(tool_input)
    if tool_name == "PatchPackage":
        return review_prompt_for_patch_package(tool_input)
    return None


def persist_review_output(tool_name: str, tool_input: dict, review_output: str) -> None:
    base_dir = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
    reviews_dir = os.path.join(base_dir, "codex_reviews")
    os.makedirs(reviews_dir, exist_ok=True)

    identifier = ""
    if tool_name == "PatchPackage":
        identifier = tool_input.get("ticket", "") or ""
    else:
        identifier = tool_input.get("file_path", "") or tool_input.get("file", "") or ""

    if not identifier:
        identifier = tool_name.lower() or "review"

    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", identifier).strip("-._") or "review"
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"{timestamp}_{slug}.txt"
    path = os.path.join(reviews_dir, filename)

    counter = 1
    while os.path.exists(path):
        path = os.path.join(reviews_dir, f"{timestamp}_{slug}_{counter}.txt")
        counter += 1

    with open(path, "w", encoding="utf-8") as f:
        f.write(review_output)


def emit(decision: str, reason: str) -> None:
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
    sys.exit(0)


def main() -> None:
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        emit("deny", f"Hook error: Failed to parse input JSON: {exc}")

    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})
    if not isinstance(tool_input, dict):
        tool_input = {}

    prompt = build_prompt(tool_name, tool_input)
    if prompt is None:
        emit("allow", f"Tool {tool_name} not configured for review")

    try:
        result = subprocess.run(
            ["codex", "exec", prompt],
            capture_output=True,
            text=True,
            timeout=1490,  # Leave 10 seconds before the 1500s hook timeout
        )
        review_output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        emit("deny", "Codex review timed out after 1490 seconds. Please try again.")
    except FileNotFoundError:
        emit(
            "deny",
            "Codex command not found. Please ensure codex is installed and in PATH.",
        )
    except Exception as exc:
        emit("deny", f"Error running codex: {exc}")

    persist_review_output(tool_name, tool_input, review_output)

    approve_match = re.search(r"^APPROVE:\s*(.*)", review_output, flags=re.MULTILINE)
    if approve_match:
        approval_msg = approve_match.group(1).strip()
        emit("allow", f"✅ Codex approved: {approval_msg}")

    reject_match = re.search(r"^REJECT:\s*(.*)", review_output, flags=re.MULTILINE)
    if reject_match:
        rejection_msg = reject_match.group(1).strip()
        emit(
            "deny",
            f"❌ Codex rejected this change:\n\n{rejection_msg}\n\nPlease revise based on the feedback above.",
        )

    emit(
        "deny",
        "⚠️  Codex response format unexpected. Expected APPROVE: or REJECT: prefix.\n\n"
        f"Codex output:\n{review_output}",
    )


if __name__ == "__main__":
    main()
