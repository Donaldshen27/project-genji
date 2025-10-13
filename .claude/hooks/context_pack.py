#!/usr/bin/env python3
import json, sys, os, re, pathlib, itertools

def last_ticket_from_text(s: str):
    m = re.findall(r'\b[A-Z]{2,}-\d+\b', s)
    return m[-1] if m else None

def read(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def tail_lines(path, n=50):
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception:
        return ""

inp = json.load(sys.stdin)
proj = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
transcript = inp.get("transcript_path", "")
tool = inp.get("tool_name","")
tool_input = inp.get("tool_input",{})
task_text = ""
for k in ("prompt","task","content","text","args"):
    v = tool_input.get(k)
    if isinstance(v, str):
        task_text = v
        break

ticket = None
if task_text:
    ticket = last_ticket_from_text(task_text)

# Fallback: scan transcript tail for a ticket pattern
if not ticket and transcript and os.path.exists(transcript):
    ticket = last_ticket_from_text(tail_lines(transcript, 500))

# Load tickets/work_items.json
tickets_path = os.path.join(proj, "tickets", "work_items.json")
tickets = {}
if os.path.exists(tickets_path):
    with open(tickets_path, "r", encoding="utf-8") as f:
        tickets = json.load(f)

target_file = None
ticket_slice = None

def walk_items(tickets_json):
    for mod in tickets_json.get("modules", []):
        for it in mod.get("items", []):
            yield mod.get("file"), it

if ticket and tickets:
    for fpath, it in walk_items(tickets):
        if it.get("id") == ticket:
            target_file = fpath
            ticket_slice = it
            break

# If still unknown, pick first ticket as fallback
if not ticket_slice:
    for fpath, it in walk_items(tickets):
        target_file, ticket, ticket_slice = fpath, it.get("id"), it
        break

repo_map = read(os.path.join(proj, "repo_map.md"))
summary_tail = tail_lines(os.path.join(proj, "summary", "ledger.jsonl"), 20)

file_region = ""
if target_file and os.path.exists(os.path.join(proj, target_file)):
    # Simplified region: first 160 lines
    with open(os.path.join(proj, target_file), "r", encoding="utf-8") as f:
        file_region = "".join(list(itertools.islice(f, 160)))

pack = {
  "ticket": ticket,
  "target_file": target_file,
  "ticket_slice": ticket_slice,
  "contracts": os.listdir(os.path.join(proj, "contracts")) if os.path.exists(os.path.join(proj, "contracts")) else [],
  "repo_map_excerpt": repo_map[:4000],
  "summary_excerpt": summary_tail,
  "file_region_excerpt": file_region
}

os.makedirs("/tmp", exist_ok=True)
with open("/tmp/context.json", "w", encoding="utf-8") as f:
    json.dump(pack, f, ensure_ascii=False, indent=2)

# success
sys.exit(0)
