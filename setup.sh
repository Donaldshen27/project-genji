#!/usr/bin/env bash
set -euo pipefail

mkdir -p patches tickets contracts summary orchestrator

if [ ! -f tickets/work_items.json ]; then
  printf '{}\n' > tickets/work_items.json
fi

touch summary/ledger.jsonl

if [ ! -f repo_map.md ]; then
  echo "# Repo Map" > repo_map.md
fi

if [ ! -f orchestrator/budgets.yaml ]; then
  cat > orchestrator/budgets.yaml <<'YAML'
planner:    {max_tokens: 16000}
skeletoner: {max_tokens: 14000}
implementer:{max_tokens: 12000}
integrator: {max_tokens: 10000}

YAML
fi

# ensure hook scripts stay executable without failing when hooks are missing
chmod +x .claude/hooks/* 2>/dev/null || true

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git init
fi

# create the initial commit only when the repo has no commits yet
if git rev-parse --is-inside-work-tree >/dev/null 2>&1 && ! git rev-parse HEAD >/dev/null 2>&1; then
  git add .
  git commit -m "Initialize Claude Code hooks and scaffold"
fi
