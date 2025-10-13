#!/usr/bin/env python3
import json, sys, os, subprocess, tempfile, shlex, datetime

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
        }
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

with open(fp, "r", encoding="utf-8") as f:
    pkg = json.load(f)

ticket = pkg.get("ticket","").strip() if isinstance(pkg.get("ticket"), str) else ""
patch_main = pkg.get("patch_unified","")
tests = pkg.get("tests", [])
review_metadata = {}
for key in ("description", "context", "notes", "summary"):
    value = pkg.get(key)
    if isinstance(value, str) and value.strip():
        review_metadata[key] = value.strip()

if not ticket:
    print(json.dumps({
      "decision": "block",
      "reason": "Patch package must include a non-empty 'ticket' field (e.g., 'Ticket: KEY-123').",
      "hookSpecificOutput": {
        "hookEventName": "PostToolUse",
        "additionalContext": "Add a ticket identifier to the patch package metadata."
      }
    }))
    sys.exit(0)

if not patch_main:
    print(json.dumps({
      "decision": "block",
      "reason": "Patch package is missing 'patch_unified'. Provide a unified diff."
    }))
    sys.exit(0)

branch = current_branch()
protected_branches = {"main", "master", "HEAD"}
if not branch or branch in protected_branches:
    reason = (
      "Patch integration requires an active feature branch. "
      "Create one with `git checkout -b feature/<ticket-slug>` and re-run."
    )
    print(json.dumps({
      "decision": "block",
      "reason": reason,
      "hookSpecificOutput": {
        "hookEventName": "PostToolUse",
        "additionalContext": reason
      }
    }))
    sys.exit(0)

main_path = write_temp(patch_main)
rc, out, err = run(f"git apply --check {shlex.quote(main_path)}")
if rc != 0:
    ctx = (err or out)[:500]
    print(json.dumps({
      "decision": "block",
      "reason": f"The patch failed to apply: {ctx}",
      "hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": ctx}
    }))
    sys.exit(0)

# Apply main patch
rc, out, err = run(f"git apply {shlex.quote(main_path)}", check=False)
# Apply test patches
test_paths = []
for t in tests:
    tp = t.get("patch_unified")
    if tp:
        pth = write_temp(tp)
        test_paths.append(pth)
        rc2, out2, err2 = run(f"git apply --check {shlex.quote(pth)}")
        if rc2 != 0:
            ctx = (err2 or out2)[:500]
            print(json.dumps({
              "decision": "block",
              "reason": f"A test patch failed to apply: {ctx}",
              "hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": ctx}
            }))
            sys.exit(0)
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
      "when": datetime.datetime.utcnow().isoformat() + "Z"
    }
    os.makedirs(os.path.join(proj, "patches"), exist_ok=True)
    cap_path = os.path.join(proj, "patches", f"{ticket}.failure.json")
    with open(cap_path, "w", encoding="utf-8") as f:
        json.dump(capsule, f, ensure_ascii=False, indent=2)
    print(json.dumps({
      "decision": "block",
      "reason": "Some tests failed; see the failure capsule.",
      "hookSpecificOutput": {
        "hookEventName": "PostToolUse",
        "additionalContext": f"Failure capsule written to {cap_path}."
      }
    }))
    sys.exit(0)

# Codex review of combined patch package
ok, review_msg = run_codex_review(proj, ticket, patch_main, review_metadata)
if not ok:
    print(json.dumps({
      "decision": "block",
      "reason": review_msg,
      "hookSpecificOutput": {
        "hookEventName": "PostToolUse",
        "additionalContext": review_msg
      }
    }))
    sys.exit(0)

# Commit on success
run("git add -A || true", check=False)
run(f"git commit -m 'Integrate {ticket} via patch package' || true", check=False)

# Append ledger entry
os.makedirs(os.path.join(proj, "summary"), exist_ok=True)
with open(os.path.join(proj, "summary", "ledger.jsonl"), "a", encoding="utf-8") as f:
    f.write(json.dumps({
      "ticket": ticket,
      "status": "pass",
      "diffstat": ""  # keep simple in starter
    }) + "\n")

# Optional: notify (stdout is fine)
print(f"Patch {ticket} applied and tests passed.")
sys.exit(0)
