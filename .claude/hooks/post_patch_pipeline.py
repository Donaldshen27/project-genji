#!/usr/bin/env python3
import datetime
import json
import os
import shlex
import subprocess
import sys
import tempfile


def run(cmd, check=True):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def write_temp(content, suffix=".patch"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def current_branch():
    rc, out, err = run("git rev-parse --abbrev-ref HEAD")
    if rc != 0:
        return None
    return out.strip()


def run_codex_review(project_dir, ticket, diff_text, metadata=None):
    review_hook = os.path.join(project_dir, ".claude", "hooks", "codex-review-hook.py")
    if not os.path.isfile(review_hook):
        return False, f"Codex review hook not found at {review_hook}."
    if not os.access(review_hook, os.X_OK):
        return False, f"Codex review hook at {review_hook} is not executable."

    meta_payload = metadata or {}
    payload = {
        "tool_name": "PatchPackage",
        "tool_input": {
            "ticket": ticket,
            "diff": diff_text,
            "metadata": meta_payload,
        },
    }

    try:
        proc = subprocess.run(
            [review_hook],
            input=json.dumps(payload),
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return False, f"Codex review hook failed to run: {exc}"

    output = (proc.stdout or "").strip()
    if not output:
        return False, "Codex review hook returned no output."

    try:
        resp = json.loads(output)
    except json.JSONDecodeError as exc:
        return False, f"Codex review hook output was not valid JSON: {exc}"

    hook_out = resp.get("hookSpecificOutput", {})
    decision = hook_out.get("permissionDecision")
    reason = hook_out.get("permissionDecisionReason", "")

    if decision == "allow":
        return True, reason

    if not reason:
        reason = "Codex review rejected this patch package."
    return False, reason


def block(reason, context=None):
    payload = {
        "decision": "block",
        "reason": reason,
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
        },
    }
    if context:
        payload["hookSpecificOutput"]["additionalContext"] = context
    print(json.dumps(payload))
    sys.exit(0)


try:
    hook_in = json.load(sys.stdin)
except Exception:
    hook_in = {}

proj = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
ti = hook_in.get("tool_input", {})
fp = ti.get("file_path") or ti.get("path") or ""

# Only act on patch packages under patches/
if not fp or "patches/" not in fp or not fp.endswith(".json"):
    sys.exit(0)

with open(fp, encoding="utf-8") as f:
    pkg = json.load(f)

ticket = pkg.get("ticket", "").strip() if isinstance(pkg.get("ticket"), str) else ""
files_raw = pkg.get("files")
tests_raw = pkg.get("tests")
review_metadata = {}
for key in ("description", "context", "notes", "summary"):
    value = pkg.get(key)
    if isinstance(value, str) and value.strip():
        review_metadata[key] = value.strip()

if not ticket:
    block(
        "Patch package must include a non-empty 'ticket' field (e.g., 'Ticket: KEY-123').",
        "Add a ticket identifier to the patch package metadata.",
    )


def ensure_list(value, label):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    block(f"Patch package field '{label}' must be an array.")


files_list = ensure_list(files_raw, "files")
tests_list = ensure_list(tests_raw, "tests")

snapshot_files = [entry for entry in files_list if isinstance(entry, dict) and "content" in entry]
snapshot_tests = [entry for entry in tests_list if isinstance(entry, dict) and "content" in entry]
legacy_file_entries = [
    entry for entry in files_list if isinstance(entry, dict) and "patch_unified" in entry
]
legacy_test_entries = [
    entry for entry in tests_list if isinstance(entry, dict) and "patch_unified" in entry
]
has_snapshot_entries = bool(snapshot_files or snapshot_tests)

branch = current_branch()
protected_branches = {"main", "master", "HEAD"}
if not branch or branch in protected_branches:
    reason = (
        "Patch integration requires an active feature branch. "
        "Create one with `git checkout -b feature/<ticket-slug>` and re-run."
    )
    block(reason, reason)

diff_text = ""
if has_snapshot_entries:
    if legacy_file_entries or legacy_test_entries:
        block("Patch package cannot mix full-file snapshots with 'patch_unified' entries.")
    if not snapshot_files:
        block("Patch package must include at least one 'files' entry with full-file 'content'.")

    def collect_snapshot_entries(entries, label):
        collected = []
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                block(f"Each entry in '{label}' must be an object containing 'path' and 'content'.")
            raw_path = entry.get("path") or entry.get("file")
            if not isinstance(raw_path, str) or not raw_path.strip():
                block(f"Entry {idx} in '{label}' is missing a valid 'path'.")
            if os.path.isabs(raw_path):
                block(f"Entry {idx} in '{label}' must use a repository-relative path.")
            rel_path = os.path.normpath(raw_path.strip())
            if rel_path.startswith(".."):
                block(f"Entry {idx} in '{label}' attempts to escape the repository root.")
            content = entry.get("content")
            if not isinstance(content, str):
                block(
                    f"Entry {idx} in '{label}' must include string 'content' with the full file text."
                )
            abs_path = os.path.join(proj, rel_path)
            collected.append((rel_path, abs_path, content))
        return collected

    files_to_write = collect_snapshot_entries(snapshot_files, "files")
    tests_to_write = collect_snapshot_entries(snapshot_tests, "tests")

    changes_made = False
    for rel_path, abs_path, content in files_to_write + tests_to_write:
        parent_dir = os.path.dirname(abs_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        existing_text = None
        try:
            with open(abs_path, encoding="utf-8") as existing_file:
                existing_text = existing_file.read()
        except FileNotFoundError:
            existing_text = None
        if existing_text != content:
            with open(abs_path, "w", encoding="utf-8") as out_file:
                out_file.write(content)
            changes_made = True

    if not changes_made:
        block(
            "Patch package produced no changes; ensure the provided 'content' reflects your edits."
        )

else:
    patch_main = pkg.get("patch_unified", "")
    if not patch_main:
        block("Patch package is missing 'patch_unified'. Provide a unified diff.")
    if snapshot_files or snapshot_tests:
        block("Patch package mixing 'patch_unified' with 'content' entries is not supported.")

    main_path = write_temp(patch_main)
    rc, out, err = run(f"git apply --check {shlex.quote(main_path)}")
    if rc != 0:
        ctx = (err or out)[:500]
        block(f"The patch failed to apply: {ctx}", ctx)

    run(f"git apply {shlex.quote(main_path)}", check=False)

    legacy_tests = []
    for entry in tests_list:
        if not isinstance(entry, dict) or "patch_unified" not in entry:
            block("Legacy patch packages must supply 'patch_unified' for each test entry.")
        legacy_tests.append(entry)

    for t in legacy_tests:
        tp = t.get("patch_unified")
        if tp:
            pth = write_temp(tp)
            rc2, out2, err2 = run(f"git apply --check {shlex.quote(pth)}")
            if rc2 != 0:
                ctx = (err2 or out2)[:500]
                block(f"A test patch failed to apply: {ctx}", ctx)
            run(f"git apply {shlex.quote(pth)}", check=False)

# Formatting & lint (best-effort)
run("ruff check --fix . || true", check=False)
run("black . || true", check=False)

# Run fast tests
rc, out, err = run(f"pytest -q -k '{ticket}' --maxfail=1 || true", check=False)
if "FAILED" in out or rc != 0:
    # Create a small failure capsule
    capsule = {
        "ticket": ticket,
        "failed_test_excerpt": (out + "\n" + err)[:600],
        "when": datetime.datetime.utcnow().isoformat() + "Z",
    }
    os.makedirs(os.path.join(proj, "patches"), exist_ok=True)
    cap_path = os.path.join(proj, "patches", f"{ticket}.failure.json")
    with open(cap_path, "w", encoding="utf-8") as f:
        json.dump(capsule, f, ensure_ascii=False, indent=2)
    block("Some tests failed; see the failure capsule.", f"Failure capsule written to {cap_path}.")

# Capture diff for Codex review (after formatters/tests)
rc_diff, diff_out, diff_err = run("git diff")
if rc_diff == 0:
    diff_text = diff_out
else:
    diff_text = ""

# Codex review of combined patch package
ok, review_msg = run_codex_review(proj, ticket, diff_text, review_metadata)
if not ok:
    block(review_msg, review_msg)

# Commit on success
run("git add -A || true", check=False)
run(f"git commit -m 'Integrate {ticket} via patch package' || true", check=False)

# Append ledger entry
os.makedirs(os.path.join(proj, "summary"), exist_ok=True)
with open(os.path.join(proj, "summary", "ledger.jsonl"), "a", encoding="utf-8") as f:
    f.write(
        json.dumps(
            {
                "ticket": ticket,
                "status": "pass",
                "diffstat": "",  # keep simple in starter
            }
        )
        + "\n"
    )

# Optional: notify (stdout is fine)
print(f"Package {ticket} applied and tests passed.")
sys.exit(0)
