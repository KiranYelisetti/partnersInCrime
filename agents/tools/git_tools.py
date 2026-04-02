"""
Git tools for agents.
All operations run in PROJECT_ROOT.
"""
import subprocess
from pathlib import Path
from langchain_core.tools import tool
from config import PROJECT_ROOT, COMMAND_TIMEOUT


def _run_git(*args: str) -> str:
    """Run a git command in PROJECT_ROOT and return output."""
    cmd = ["git"] + list(args)
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            error = result.stderr.strip()
            return f"Git error (exit {result.returncode}):\n{error}\n{output}"
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Git command timed out after {COMMAND_TIMEOUT}s: {' '.join(cmd)}"
    except FileNotFoundError:
        return "Error: git is not installed or not in PATH"


@tool
def git_status() -> str:
    """Show the current git status (modified, staged, untracked files)."""
    return _run_git("status", "--short", "--branch")


@tool
def git_diff(file_path: str = "") -> str:
    """
    Show unstaged changes. If file_path is provided, show diff for that file only.
    """
    args = ["diff", "--stat"]
    if file_path:
        # Also show the actual diff for the specific file
        full_diff = _run_git("diff", file_path)
        stat = _run_git("diff", "--stat", file_path)
        return f"Stats:\n{stat}\n\nDiff:\n{full_diff}"
    return _run_git(*args)


@tool
def git_diff_staged() -> str:
    """Show staged (added but not committed) changes."""
    return _run_git("diff", "--cached", "--stat")


@tool
def git_add(files: str = ".") -> str:
    """
    Stage files for commit. Pass specific file paths (comma-separated) or '.' for all.
    """
    if files == ".":
        return _run_git("add", ".")
    file_list = [f.strip() for f in files.split(",")]
    return _run_git("add", *file_list)


@tool
def git_commit(message: str) -> str:
    """Commit staged changes with the given message."""
    if not message.strip():
        return "Error: Commit message cannot be empty."
    return _run_git("commit", "-m", message)


@tool
def git_log(count: int = 10) -> str:
    """Show the last N commits with hash, author, date, and message."""
    return _run_git(
        "log",
        f"-{count}",
        "--oneline",
        "--graph",
        "--decorate",
    )


@tool
def git_branch_list() -> str:
    """List all branches, highlighting the current one."""
    return _run_git("branch", "-a")


@tool
def git_branch_create(name: str) -> str:
    """Create a new branch and switch to it."""
    result = _run_git("checkout", "-b", name)
    return result


@tool
def git_checkout(branch: str) -> str:
    """Switch to an existing branch."""
    return _run_git("checkout", branch)


@tool
def git_init() -> str:
    """Initialize a new git repository in PROJECT_ROOT (if not already initialized)."""
    return _run_git("init")


# Export all git tools
GIT_TOOLS = [
    git_status, git_diff, git_diff_staged, git_add, git_commit,
    git_log, git_branch_list, git_branch_create, git_checkout, git_init,
]
