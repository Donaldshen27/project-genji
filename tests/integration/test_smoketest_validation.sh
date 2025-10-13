#!/bin/bash
# Integration tests for SMK-001-001: smoketest.md validation
# These tests verify the file exists, has correct content, and does not affect other files.

set -e

# Compute project root dynamically from script location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET_FILE="$PROJECT_ROOT/smoketest.md"

echo Test 1: Verify file exists
if [ ! -f "$TARGET_FILE" ]; then
    echo FAIL: smoketest.md does not exist
    exit 1
fi
echo PASS: File exists

echo Test 2: Verify exact content match
EXPECTED='new workflow smoketest'
ACTUAL=$(cat "$TARGET_FILE")
if [ "$ACTUAL" != "$EXPECTED" ]; then
    echo FAIL: Content mismatch
    exit 1
fi
echo PASS: Content matches exactly

echo Test 3: Verify file ends with exactly one newline
# Use hexdump to check last byte is newline and second-to-last is not
LAST_BYTE=$(tail -c 1 "$TARGET_FILE" | od -An -tx1 | tr -d ' ')
if [ "$LAST_BYTE" != "0a" ]; then
    echo FAIL: File does not end with newline
    exit 1
fi
LAST_TWO=$(tail -c 2 "$TARGET_FILE" | od -An -tx1 | tr -d ' ')
if [ "$LAST_TWO" = "0a0a" ]; then
    echo FAIL: File has multiple trailing newlines
    exit 1
fi
echo PASS: File ends with exactly one newline

echo Test 4: Verify only expected files in git status
cd "$PROJECT_ROOT"
GIT_STATUS=$(git status --porcelain)
if [ -z "$GIT_STATUS" ]; then
    echo PASS: Git status clean
else
    # Check each line for exact path match and exit immediately on mismatch
    ALLOWED_FILES="smoketest.md tests/integration/test_smoketest_validation.sh patches/SMK-001-001.json"
    while IFS= read -r line; do
        # Extract filename from porcelain format (skip first 3 chars)
        filepath=$(echo "$line" | cut -c4-)
        # Check if this file is in allowed list
        FOUND=0
        for allowed in $ALLOWED_FILES; do
            if [ "$filepath" = "$allowed" ]; then
                FOUND=1
                break
            fi
        done
        if [ "$FOUND" -eq 0 ]; then
            echo FAIL: Unexpected file modified: $filepath
            echo Full git status:
            echo "$GIT_STATUS"
            exit 1
        fi
    done < <(echo "$GIT_STATUS")
    echo PASS: Only expected files in git status
fi

echo All integration tests passed for SMK-001-001
