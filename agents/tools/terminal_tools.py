"""
Terminal/shell tools for agents.
Sandboxed with command allowlist and timeout.
"""
import subprocess
import shlex
import platform
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

    try:
        # Use shell=True on Windows for proper command resolution
        use_shell = platform.system() == "Windows"
        result = subprocess.run(
            cmd,
            shell=use_shell,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

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
        if result.returncode != 0:
            output_parts.insert(0, f"Exit code: {result.returncode}")

        return "\n\n".join(output_parts) if output_parts else "(no output, exit code 0)"

    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s: {cmd}"
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
    return _run(command)


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


@tool
def npm_run(script: str) -> str:
    """Run an npm script defined in package.json.
    Args:
        script: The script name (e.g. "build", "dev", "test", "lint")
    """
    safe = script.strip().split()[0]  # Take only the script name
    if not all(c.isalnum() or c in "-_:" for c in safe):
        return f"Error: Invalid script name: {safe}"
    return _run(f"npm run {safe}", timeout=120)


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
