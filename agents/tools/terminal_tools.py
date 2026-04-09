"""
Terminal/shell tools for agents.
Sandboxed with command allowlist and timeout.
"""
import subprocess
import shlex
import platform
import re
from pathlib import Path
from langchain_core.tools import tool
from config import (
    PROJECT_ROOT, COMMAND_TIMEOUT, ALLOWED_COMMANDS,
    MAX_COMMAND_OUTPUT,
)


def _get_command_name(cmd_str: str) -> str:
    """Extract the base command name from a command string."""
    try:
        parts = shlex.split(cmd_str)
    except ValueError:
        parts = cmd_str.split()
    if not parts:
        return ""
    # Get just the executable name, not the full path
    return Path(parts[0]).stem.lower()


# ── Denylist: long-running / interactive commands that would hang agents ─
# These are valid commands but start servers/watchers that never exit.
# Agents must use `npm run build` (one-shot) instead of `npm run dev`.
_DENY_PATTERNS = [
    # Dev servers — start HTTP servers that never exit
    r"\bnpm\s+(run\s+)?(dev|start|serve|watch)\b",
    r"\bnpm\s+run\s+dev(:|$|\s)",
    r"\bnext\s+(dev|start)\b",
    r"^\s*vite\s*$",                 # Plain `vite` alone starts dev server
    r"\bvite\s+dev\b",
    r"\bvite\s+preview\b",
    r"\bvite\s+serve\b",
    r"\bwebpack\s+(serve|--watch|-w\b)",
    r"\btsc\s+(--watch|-w\b)",
    r"\bnodemon\b",
    r"\bnpx\s+(next|vite|nodemon)\s+(dev|start|serve|watch)",
    r"\bnpx\s+serve\b",
    # Interactive prompts / REPLs
    r"\bnpm\s+init(\s|$)(?!.*-y)",   # `npm init` without -y waits for input
    r"\bnode\s*$",                    # Plain `node` starts REPL
    r"\bpython\s*$",                  # Plain `python` starts REPL
    # Infinite loops / long-running watchers
    r"\btail\s+-f\b",
    r"\bwatch\b",
]


def _is_denied_command(cmd: str) -> str:
    """Return a reason if the command is on the denylist, empty string if OK."""
    for pat in _DENY_PATTERNS:
        if re.search(pat, cmd, re.IGNORECASE):
            return pat
    return ""


def _run(cmd: str, timeout: int = None, cwd: str = None) -> str:
    """Run a shell command with safety checks."""
    timeout = timeout or COMMAND_TIMEOUT
    work_dir = cwd or str(PROJECT_ROOT)

    # Safety: check command is in allowlist
    cmd_name = _get_command_name(cmd)
    if cmd_name not in ALLOWED_COMMANDS:
        return (
            f"Error: Command '{cmd_name}' is not in the allowed commands list.\n"
            f"Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}\n"
            f"Add it to ALLOWED_COMMANDS in .env if you trust this command."
        )

    # Safety: block long-running commands (dev servers, REPLs, watchers)
    denied = _is_denied_command(cmd)
    if denied:
        return (
            f"Error: Command BLOCKED — '{cmd}' is a long-running / interactive command "
            f"that would hang the agent loop (matched denylist pattern: {denied}).\n"
            f"For build verification use 'npm run build' (one-shot compile), "
            f"NOT 'npm run dev' (starts a dev server that never exits).\n"
            f"For type-checking use 'npx tsc --noEmit' (not 'tsc --watch')."
        )

    try:
        is_windows = platform.system() == "Windows"

        # On Windows, use CREATE_NEW_PROCESS_GROUP so we can kill the whole tree
        # if timeout fires. Otherwise npm.cmd → node child survives and its pipes
        # keep subprocess.communicate() hanging forever.
        popen_kwargs = {
            "cwd": work_dir,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "shell": is_windows,
            "text": True,
        }
        if is_windows:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True  # POSIX: new process group

        proc = subprocess.Popen(cmd, **popen_kwargs)

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            # Hard kill the entire process tree.
            if is_windows:
                # taskkill /T kills the whole tree including node.exe children
                subprocess.run(
                    f"taskkill /F /T /PID {proc.pid}",
                    shell=True, capture_output=True, timeout=10,
                )
            else:
                import os, signal
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
            # Reap the process so pipes close
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except Exception:
                stdout, stderr = "", ""
            return (
                f"Error: Command timed out after {timeout}s and was force-killed: {cmd}\n"
                f"If this was a dev server, use 'npm run build' instead. "
                f"Agents should never start long-running processes."
            )

        stdout = (stdout or "").strip()
        stderr = (stderr or "").strip()

        # Truncate long output
        if len(stdout) > MAX_COMMAND_OUTPUT:
            stdout = f"...(truncated, showing last {MAX_COMMAND_OUTPUT} chars)...\n" + stdout[-MAX_COMMAND_OUTPUT:]
        if len(stderr) > MAX_COMMAND_OUTPUT:
            stderr = f"...(truncated)...\n" + stderr[-MAX_COMMAND_OUTPUT:]

        output_parts = []
        if stdout:
            output_parts.append(f"stdout:\n{stdout}")
        if stderr:
            output_parts.append(f"stderr:\n{stderr}")
        if returncode != 0:
            output_parts.insert(0, f"Exit code: {returncode}")

        return "\n\n".join(output_parts) if output_parts else "(no output, exit code 0)"

    except FileNotFoundError:
        return f"Error: Command not found: {cmd}"
    except Exception as e:
        return f"Error running command: {e}"


@tool
def run_command(command: str) -> str:
    """
    Run a shell command in the project directory.
    The command must be in the allowed commands list (see .env).
    Returns stdout/stderr and exit code.

    Examples:
        run_command("python --version")
        run_command("npm run dev")
        run_command("pytest tests/ -v")
    """
    result = _run(command)

    # Filter build errors to only show agent's own files
    if "npm run build" in command and "Exit code:" in result:
        from tools.hooks import get_current_agent
        agent = get_current_agent()
        if agent:
            result = _filter_build_errors(result, agent)

    return result


@tool
def run_command_with_timeout(command: str, timeout_seconds: int = 60) -> str:
    """
    Run a shell command with a custom timeout (useful for builds, installs).
    """
    return _run(command, timeout=min(timeout_seconds, 300))  # Cap at 5 minutes


@tool
def run_python(script_path: str) -> str:
    """
    Run a Python script in the project directory.
    """
    path = Path(script_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        return f"Error: Script not found: {path}"
    if path.suffix != ".py":
        return f"Error: Not a Python file: {path}"

    return _run(f"python \"{path}\"")


@tool
def install_package(package_name: str) -> str:
    """Install a Python package using pip."""
    # Basic safety — no shell injection via package name
    safe_name = package_name.strip().split()[0]  # Take only first word
    if not all(c.isalnum() or c in "-_.[],><=!" for c in safe_name):
        return f"Error: Invalid package name: {safe_name}"
    return _run(f"pip install {safe_name}", timeout=120)


@tool
def run_tests(test_path: str = "tests/", verbose: bool = True) -> str:
    """
    Run pytest on the given path.
    """
    cmd = f"pytest {test_path}"
    if verbose:
        cmd += " -v"
    return _run(cmd, timeout=120)


@tool
def npm_install(packages: str = "") -> str:
    """Install Node.js packages using npm.
    Args:
        packages: Space-separated package names (e.g. "express typescript @types/node").
                  Leave empty for plain 'npm install' from package.json.
    """
    if packages.strip():
        # Basic safety — only allow reasonable package name characters
        for pkg in packages.split():
            if not all(c.isalnum() or c in "-_@/.<>=^~" for c in pkg):
                return f"Error: Invalid package name: {pkg}"
        return _run(f"npm install {packages}", timeout=120)
    else:
        return _run("npm install", timeout=120)


def _filter_build_errors(output: str, agent_name: str) -> str:
    """
    Filter build output to only show errors in files the current agent owns.
    If ALL errors are in other agents' files, return a short "pre-existing" message
    so the agent doesn't waste iterations trying to fix them.
    """
    import re
    from tools.hooks import get_current_agent, AGENT_WRITE_SCOPES

    if not agent_name or "Error" not in output and "error" not in output:
        return output

    scopes = AGENT_WRITE_SCOPES.get(agent_name)
    if scopes is None or len(scopes) == 0:
        return output  # Unrestricted agent — show everything

    includes = [s for s in scopes if not s.startswith("!")]
    excludes = [s[1:] for s in scopes if s.startswith("!")]

    def _is_my_file(filepath: str) -> bool:
        """Check if a file path falls within this agent's write scope."""
        fp = filepath.replace("\\", "/")
        for exc in excludes:
            if fp.startswith(exc):
                return False
        for inc in includes:
            if fp.startswith(inc):
                return True
        return False

    # Extract error lines referencing files (TypeScript pattern: ./src/path/file.ts)
    lines = output.split("\n")
    my_errors = []
    other_errors = 0
    for line in lines:
        # Match TS error patterns like "./src/app/api/payment/verify.ts(3,30)"
        # or "src/app/api/payment/verify.ts:3:30"
        match = re.search(r'[./]*((src|app|pages|components)/\S+?\.(ts|tsx|js|jsx))', line)
        if match:
            filepath = match.group(1).lstrip("./")
            if _is_my_file(filepath):
                my_errors.append(line)
            else:
                other_errors += 1
        else:
            # Non-file-reference lines (summaries, etc.) — keep them
            my_errors.append(line)

    if other_errors > 0 and not any("error" in l.lower() for l in my_errors if re.search(r'src/\S+\.(ts|tsx)', l)):
        return (
            f"Build has errors, but ALL {other_errors} error(s) are in OTHER agents' files "
            f"(outside your scope: {includes}). These are pre-existing issues you cannot fix.\n"
            f"YOUR code compiled successfully. Proceed with task_done."
        )

    if other_errors > 0:
        filtered = "\n".join(my_errors)
        return (
            f"{filtered}\n\n"
            f"NOTE: {other_errors} additional error(s) in other agents' files were hidden. "
            f"Focus only on errors in YOUR files."
        )

    return output


@tool
def npm_run(script: str) -> str:
    """Run a one-shot npm script defined in package.json.
    Use this for 'build', 'test', 'lint', 'typecheck' — scripts that run once and exit.
    Args:
        script: The script name (e.g. "build", "test", "lint")
    NOTE: Long-running scripts like 'dev', 'start', 'serve', 'watch' are BLOCKED.
    They start servers that never exit and would hang the agent loop.
    For build verification use 'build'.
    """
    safe = script.strip().split()[0]  # Take only the script name
    if not all(c.isalnum() or c in "-_:" for c in safe):
        return f"Error: Invalid script name: {safe}"
    # Block long-running scripts — same denylist as run_command
    forbidden = {"dev", "start", "serve", "watch"}
    if safe.lower() in forbidden or any(safe.lower().startswith(f + ":") for f in forbidden):
        return (
            f"Error: npm script '{safe}' is blocked — it starts a long-running process "
            f"that never exits and would hang the agent loop. "
            f"For build verification use npm_run('build'). "
            f"For tests use npm_run('test')."
        )
    result = _run(f"npm run {safe}", timeout=120)

    # For build scripts, filter errors to only show agent's own files
    if safe.lower() == "build" and "Exit code:" in result:
        from tools.hooks import get_current_agent
        agent = get_current_agent()
        if agent:
            result = _filter_build_errors(result, agent)

    return result


@tool
def npx_command(command: str) -> str:
    """Run an npx command (e.g. 'create-next-app', 'tsc --noEmit', 'prisma generate').
    Args:
        command: The npx command and arguments
    """
    # create-next-app and similar scaffolding tools need more time
    timeout = 300 if "create-" in command else 180
    return _run(f"npx {command}", timeout=timeout)


# Export all terminal tools
TERMINAL_TOOLS = [
    run_command, run_command_with_timeout, run_python,
    install_package, run_tests,
    npm_install, npm_run, npx_command,
]
