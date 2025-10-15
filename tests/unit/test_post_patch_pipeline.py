import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = REPO_ROOT / ".claude" / "hooks" / "post_patch_pipeline.py"


def _init_git_repo(repo_path: Path) -> None:
    init_cmd = ["git", "init", "-b", "feature/init-branch"]
    result = subprocess.run(init_cmd, cwd=repo_path, capture_output=True, text=True)
    if result.returncode != 0:
        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "checkout", "-b", "feature/init-branch"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "tester@example.com"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )


def _install_codex_hook(repo_path: Path) -> None:
    hook_dir = repo_path / ".claude" / "hooks"
    hook_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hook_dir / "codex-review-hook.py"
    hook_source = (
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "try:\n"
        "    payload = json.load(sys.stdin)\n"
        "except Exception:\n"
        "    payload = {}\n"
        "response = {\n"
        '    "hookSpecificOutput": {\n'
        '        "permissionDecision": "allow",\n'
        '        "permissionDecisionReason": "approved"\n'
        "    }\n"
        "}\n"
        "print(json.dumps(response))\n"
    )
    hook_path.write_text(hook_source, encoding="utf-8")
    current_mode = hook_path.stat().st_mode
    hook_path.chmod(current_mode | stat.S_IEXEC)


def _seed_repo_with_file(repo_path: Path, rel_path: str, content: str) -> None:
    file_path = repo_path / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", rel_path], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=repo_path, check=True, capture_output=True)


def _write_package(repo_path: Path, filename: str, package: dict) -> Path:
    patches_dir = repo_path / "patches"
    patches_dir.mkdir(exist_ok=True)
    package_path = patches_dir / filename
    package_path.write_text(json.dumps(package), encoding="utf-8")
    return package_path


def _run_pipeline(repo_path: Path, package_path: Path) -> subprocess.CompletedProcess[str]:
    payload = json.dumps({"tool_input": {"file_path": str(package_path.relative_to(repo_path))}})
    env = os.environ.copy()
    env["CLAUDE_PROJECT_DIR"] = str(repo_path)
    return subprocess.run(
        ["python3", str(HOOK_PATH)],
        cwd=repo_path,
        env=env,
        text=True,
        input=payload,
        capture_output=True,
    )


def _git_log_message(repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "log", "-1", "--pretty=%B"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture()
def snapshot_repo(tmp_path: Path) -> Path:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    _init_git_repo(repo_path)
    _install_codex_hook(repo_path)
    _seed_repo_with_file(repo_path, "src/example.txt", "original\n")
    return repo_path


def test_snapshot_package_writes_files_and_commits(snapshot_repo: Path) -> None:
    package = {
        "ticket": "TEST-100",
        "description": "update file via snapshot",
        "files": [
            {
                "path": "src/example.txt",
                "content": "updated content\n",
            }
        ],
        "tests": [],
    }
    package_path = _write_package(snapshot_repo, "TEST-100.json", package)

    proc = _run_pipeline(snapshot_repo, package_path)

    assert proc.returncode == 0
    assert proc.stdout.strip() == "Package TEST-100 applied and tests passed."
    assert (snapshot_repo / "src/example.txt").read_text(encoding="utf-8") == "updated content\n"
    assert _git_log_message(snapshot_repo) == "Integrate TEST-100 via patch package"


def test_snapshot_package_without_changes_blocks(snapshot_repo: Path) -> None:
    package = {
        "ticket": "TEST-101",
        "description": "noop snapshot",
        "files": [
            {
                "path": "src/example.txt",
                "content": "original\n",
            }
        ],
        "tests": [],
    }
    package_path = _write_package(snapshot_repo, "TEST-101.json", package)
    previous_head = _git_log_message(snapshot_repo)

    proc = _run_pipeline(snapshot_repo, package_path)

    assert proc.returncode == 0
    message = json.loads(proc.stdout)
    assert message["decision"] == "block"
    assert "no changes" in message["reason"].lower()
    assert _git_log_message(snapshot_repo) == previous_head


def test_package_mixing_snapshots_and_diffs_blocks(snapshot_repo: Path) -> None:
    package = {
        "ticket": "TEST-102",
        "description": "mix snapshots and patch_unified",
        "files": [
            {
                "path": "src/example.txt",
                "content": "irrelevant\n",
                "patch_unified": "@@ -1 +1 @@\n-foo\n+bar\n",
            }
        ],
        "tests": [],
    }
    package_path = _write_package(snapshot_repo, "TEST-102.json", package)

    proc = _run_pipeline(snapshot_repo, package_path)

    assert proc.returncode == 0
    message = json.loads(proc.stdout)
    assert message["decision"] == "block"
    assert "cannot mix full-file snapshots" in message["reason"].lower()


def test_legacy_patch_package_still_applies(tmp_path: Path) -> None:
    repo_path = tmp_path / "legacy_repo"
    repo_path.mkdir()
    _init_git_repo(repo_path)
    _install_codex_hook(repo_path)
    _seed_repo_with_file(repo_path, "src/example.txt", "legacy\n")

    legacy_patch = """--- a/src/example.txt
+++ b/src/example.txt
@@ -1 +1 @@
-legacy
+modern
"""
    package = {
        "ticket": "TEST-103",
        "description": "legacy patch flow",
        "patch_unified": legacy_patch,
        "tests": [],
    }
    package_path = _write_package(repo_path, "TEST-103.json", package)

    proc = _run_pipeline(repo_path, package_path)

    assert proc.returncode == 0
    assert (repo_path / "src/example.txt").read_text(encoding="utf-8") == "modern\n"
    assert _git_log_message(repo_path) == "Integrate TEST-103 via patch package"
