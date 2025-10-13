#!/usr/bin/env bash
set -euo pipefail

# Script to remove transient artifacts created by the automated hooks so you
# can start a fresh ticket without leftover patch packages or failure capsules.

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
PATCH_DIR="$ROOT_DIR/patches"
SUMMARY_DIR="$ROOT_DIR/summary"

echo "Cleaning transient artifacts under $ROOT_DIR"

if [ -d "$PATCH_DIR" ]; then
  find "$PATCH_DIR" -maxdepth 1 -type f -name '*.json' -print -delete
fi

# Optional: remove Codex failure capsules if present (same directory)
if [ -d "$PATCH_DIR" ]; then
  find "$PATCH_DIR" -maxdepth 1 -type f -name '*.failure.json' -print -delete
fi

# Clear smoke/status notes if they exist
if [ -f "$SUMMARY_DIR/status.md" ]; then
  rm -f "$SUMMARY_DIR/status.md"
  echo "Removed $SUMMARY_DIR/status.md"
fi

echo "Workspace reset complete."
