"""
Agent Base — ReAct tool-calling execution engine.

This is the CORE of what makes the system autonomous. Every specialist agent
runs in a tool-calling loop:

  1. Receive task + context (what previous agents built, RAG memory)
  2. LLM THINKS about what to do next
  3. LLM CALLS a tool (read file, write code, run command, etc.)
  4. Engine EXECUTES the tool and feeds result back to LLM
  5. REPEAT steps 2-4 until agent calls task_done()
  6. Log result to memory, return to orchestrator

This is identical to how Claude Code works:
  Think → Act → Observe → Think → Act → Observe → ... → Done

Agents don't just generate text — they READ the project, WRITE real files,
RUN code, SEE errors, and FIX them autonomously.
"""
import json
import time
import uuid
import hashlib
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timezone

from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage,
)
from langchain_core.tools import BaseTool, tool
from rich import print as rprint
from rich.panel import Panel

from config import (
    get_llm, LLM_COOLDOWN, MAX_TOOL_ITERATIONS,
    MAX_TOOL_RESULT_CHARS, PROJECT_ROOT,
    BUILD_VERIFY_ENABLED, BUILD_VERIFY_MAX_RETRIES,
    AGENT_TIMEOUT_SECONDS, MAX_IDENTICAL_TOOL_CALLS, AGENT_LOG_DIR,
)
from state import AgentState


# ── Run ID: unique per pipeline invocation ───────────────────────
# Set once at import, reset by orchestrator at pipeline start.
_current_run_id: str = ""


def set_run_id(run_id: str):
    global _current_run_id
    _current_run_id = run_id


def get_run_id() -> str:
    return _current_run_id


# ── Agent Logger: per-agent structured audit trail ───────────────

class AgentLogger:
    """Writes structured JSONL logs per agent per run for full traceability."""

    def __init__(self, run_id: str, agent_name: str, task_id: str):
        self.run_id = run_id
        self.agent_name = agent_name
        self.task_id = task_id
        self._log_dir = AGENT_LOG_DIR / run_id
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / f"{agent_name}.jsonl"
        self._start_time = time.time()
        self.log("agent_start", {"task_id": task_id})

    def log(self, event: str, data: dict = None):
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(time.time() - self._start_time, 2),
            "run_id": self.run_id,
            "task_id": self.task_id,
            "agent": self.agent_name,
            "event": event,
            **(data or {}),
        }
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass  # Logging should never crash the agent

    def log_tool_call(self, iteration: int, tool_name: str, tool_args: dict, result: str, duration_s: float):
        self.log("tool_call", {
            "iteration": iteration,
            "tool": tool_name,
            "args_summary": _summarize_args(tool_args, max_len=200),
            "result_preview": (result or "")[:200],
            "duration_s": round(duration_s, 2),
        })

    def log_completion(self, success: bool, iterations: int, files: list, summary: str):
        self.log("agent_done", {
            "success": success,
            "iterations": iterations,
            "files_changed": files,
            "summary": summary[:300],
            "total_time_s": round(time.time() - self._start_time, 2),
        })


# ── Cycle Detector: catches repetitive tool call loops ───────────

class CycleDetector:
    """Tracks identical tool calls and detects loops.

    Returns (message, severity) from check():
      severity = "none"  — no issue
      severity = "warn"  — warn the agent, keep going
      severity = "stop"  — hard stop, the agent is in a pathological loop
    """

    def __init__(self, max_identical: int = MAX_IDENTICAL_TOOL_CALLS):
        self.max_identical = max_identical
        self._last_signature = ""
        self._consecutive_count = 0
        self._all_signatures: Dict[str, int] = {}  # signature -> count

    def _make_signature(self, tool_name: str, tool_args: dict) -> str:
        """Create a hashable signature from tool name + args."""
        args_str = json.dumps(tool_args, sort_keys=True, default=str)
        return hashlib.md5(f"{tool_name}:{args_str}".encode()).hexdigest()

    def check(self, tool_name: str, tool_args: dict) -> tuple[Optional[str], str]:
        """
        Record a tool call and check for cycles.
        Returns (message, severity) where severity is one of:
          "none" — no cycle detected
          "warn" — warn the agent to change approach
          "stop" — hard stop, the agent is in a pathological loop
        """
        sig = self._make_signature(tool_name, tool_args)

        # Track global frequency
        self._all_signatures[sig] = self._all_signatures.get(sig, 0) + 1
        total_count = self._all_signatures[sig]

        # Track consecutive identical calls
        if sig == self._last_signature:
            self._consecutive_count += 1
        else:
            self._last_signature = sig
            self._consecutive_count = 1

        # ── Hard stop: consecutive repeats beyond warning threshold ──
        if self._consecutive_count >= self.max_identical + 1:
            return (
                f"CYCLE DETECTED (HARD STOP): You called `{tool_name}` with the same "
                f"arguments {self._consecutive_count} times in a row despite being warned. "
                f"Terminating this agent step.",
                "stop",
            )

        # ── Hard stop: non-consecutive repetition past ceiling ──
        # Agent is interleaving a harmless read between failed writes to avoid
        # the consecutive counter. Stop it by total count too.
        if total_count >= self.max_identical + 4:
            return (
                f"CYCLE DETECTED (HARD STOP): You called `{tool_name}` with the same "
                f"arguments {total_count} times total — your approach is not working "
                f"and you are not making progress. Terminating this agent step.",
                "stop",
            )

        # ── Warn: consecutive cycle at warning threshold ──
        if self._consecutive_count >= self.max_identical:
            return (
                f"CYCLE DETECTED: You called `{tool_name}` with the same arguments "
                f"{self._consecutive_count} times in a row. You are stuck in a loop. "
                f"STOP repeating this action. Try a DIFFERENT approach (read the file "
                f"to see current state, or use a different tool) or call task_done "
                f"if you cannot proceed. Next repeat WILL terminate this step.",
                "warn",
            )

        # ── Warn: non-consecutive repetition approaching ceiling ──
        if total_count >= self.max_identical + 2:
            return (
                f"REPETITION WARNING: You have called `{tool_name}` with identical "
                f"arguments {total_count} times total during this task. This suggests "
                f"you are not making progress. Try a fundamentally different approach. "
                f"If you reach {self.max_identical + 4} repeats this step will terminate.",
                "warn",
            )

        return (None, "none")

    def reset(self):
        self._last_signature = ""
        self._consecutive_count = 0


# ── The "done" sentinel tool ──────────────────────────────────────
# When an agent calls this, the loop exits. This gives the LLM a clear
# way to signal "I'm finished" rather than us guessing from silence.

_last_done_result: Dict[str, Any] = {}


@tool
def task_done(summary: str, files_created: str = "", files_modified: str = "") -> str:
    """Call this when you have FINISHED the task successfully.
    Args:
        summary: What you accomplished (1-3 sentences)
        files_created: Comma-separated paths of NEW files you created
        files_modified: Comma-separated paths of EXISTING files you changed
    """
    global _last_done_result
    _last_done_result = {
        "summary": summary,
        "files_created": [f.strip() for f in files_created.split(",") if f.strip()],
        "files_modified": [f.strip() for f in files_modified.split(",") if f.strip()],
    }
    return f"Task marked complete. Summary: {summary}"


# ── LLM invocation with retry ────────────────────────────────────

def _invoke_with_retry(llm, messages, max_attempts=3, cooldown=None):
    """
    Call llm.invoke() with retry logic for transient failures.
    Handles both Ollama (GPU crashes, connection drops) and Claude API
    (rate limits, overloaded, timeouts) errors with appropriate backoff.
    """
    from config import LLM_PROVIDER

    cd = cooldown if cooldown is not None else LLM_COOLDOWN
    if cd > 0:
        time.sleep(cd)

    for attempt in range(max_attempts):
        try:
            return llm.invoke(messages)
        except Exception as e:
            error_str = str(e).lower()

            # Non-retryable errors — fail immediately
            is_fatal = any(phrase in error_str for phrase in [
                "credit balance is too low",
                "invalid api key",
                "invalid x-api-key",
                "authentication_error",
                "permission_error",
                "not_found_error",
            ])
            if is_fatal:
                rprint(f"  [red]Fatal API error (not retryable): {str(e)[:200]}[/red]")
                raise

            # Ollama transient errors
            ollama_transient = any(phrase in error_str for phrase in [
                "llama runner process has terminated",
                "connection refused", "connection reset", "connection error",
                "status code: 500", "internal server error", "model is loading",
            ])

            # Claude API transient errors
            claude_transient = any(phrase in error_str for phrase in [
                "429",                  # rate limit
                "rate limit",
                "overloaded",           # 529 overloaded
                "529",
                "timeout",              # request timeout
                "timed out",
                "server error",         # 500 internal
                "bad gateway",          # 502
                "service unavailable",  # 503
            ])

            is_transient = ollama_transient or claude_transient

            if is_transient and attempt < max_attempts - 1:
                if claude_transient and "429" in error_str or "rate limit" in error_str:
                    # Rate limits need longer backoff
                    wait = (attempt + 1) * 15
                    rprint(f"  [yellow]Claude rate limit hit, waiting {wait}s... "
                           f"({attempt + 1}/{max_attempts})[/yellow]")
                elif claude_transient:
                    wait = (attempt + 1) * 10
                    rprint(f"  [yellow]Claude API error, waiting {wait}s... "
                           f"({attempt + 1}/{max_attempts})[/yellow]")
                else:
                    wait = (attempt + 1) * 8
                    rprint(f"  [yellow]Ollama error, waiting {wait}s for recovery... "
                           f"({attempt + 1}/{max_attempts})[/yellow]")
                time.sleep(wait)
            else:
                raise


# ── Utility ───────────────────────────────────────────────────────

def _truncate(text: str, max_chars: int = None) -> str:
    """Truncate long text to save context window space."""
    limit = max_chars or MAX_TOOL_RESULT_CHARS
    if len(text) <= limit:
        return text
    half = limit // 2
    return (
        text[:half]
        + f"\n\n... ({len(text) - limit} chars truncated) ...\n\n"
        + text[-half:]
    )


def _summarize_args(args: dict, max_len: int = 80) -> str:
    """One-line summary of tool args for logging."""
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > max_len:
            s = s[:max_len] + "..."
        parts.append(f"{k}={s}")
    summary = ", ".join(parts)
    return summary[:200]


def _detect_project_framework() -> str:
    """
    Auto-detect project framework from package.json and inject the right
    patterns into every agent's task prompt. This prevents agents from
    using wrong patterns (e.g. Express in a Next.js project).
    """
    root = Path(PROJECT_ROOT)
    pkg_json = root / "package.json"

    if not pkg_json.exists():
        # Python project?
        if (root / "pyproject.toml").exists() or (root / "requirements.txt").exists():
            return (
                "\n## Project Type: Python\n"
                "Write .py files. Use pip for dependencies."
            )
        return ""

    try:
        pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""

    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    has_ts = (root / "tsconfig.json").exists() or "typescript" in deps

    # Next.js
    if "next" in deps:
        return (
            "\n## Project Type: Next.js App Router (TypeScript)\n"
            "This is a Next.js project with App Router. Follow these rules:\n\n"
            "**File structure:**\n"
            "- Pages go in `src/app/` (e.g. `src/app/dashboard/page.tsx`)\n"
            "- API routes go in `src/app/api/` as `route.ts` files "
            "(e.g. `src/app/api/auth/route.ts`)\n"
            "- Shared components go in `src/components/`\n"
            "- Library/utility code goes in `src/lib/`\n"
            "- Database models go in `src/models/` or `src/lib/models/`\n\n"
            "**API routes pattern:**\n"
            "```typescript\n"
            "// src/app/api/example/route.ts\n"
            "import { NextRequest, NextResponse } from 'next/server';\n"
            "export async function GET(request: NextRequest) {\n"
            "  return NextResponse.json({ data: 'hello' });\n"
            "}\n"
            "export async function POST(request: NextRequest) {\n"
            "  const body = await request.json();\n"
            "  return NextResponse.json({ success: true });\n"
            "}\n"
            "```\n\n"
            "**Client components** that use hooks (useState, useEffect) MUST have "
            "the string `\"use client\";` (WITH QUOTES) as the VERY FIRST LINE of the file. "
            "Example:\n"
            "```typescript\n"
            "\"use client\";\n"
            "import { useState } from 'react';\n"
            "```\n"
            "WRONG: `use client` (no quotes), `/* use client */` (comment), `'use client'` at line 2+.\n"
            "The directive MUST be a quoted string on line 1.\n\n"
            "**DO NOT use Express patterns** (no `import express`, no `Request/Response` "
            "from express, no `app.get()`). Use Next.js `NextRequest`/`NextResponse`.\n\n"
            "**TypeScript:** Write ALL code as .ts/.tsx files with import/export syntax."
        )

    # Vite / React
    if "vite" in deps:
        ts_note = "Write .ts/.tsx files with import/export." if has_ts else ""
        return (
            f"\n## Project Type: Vite + React\n"
            f"This is a Vite project. Components in src/components/, pages in src/pages/.\n"
            f"{ts_note}"
        )

    # Express
    if "express" in deps:
        ts_note = "Write .ts files with import/export." if has_ts else ""
        return (
            f"\n## Project Type: Express.js\n"
            f"This is an Express server. Routes in src/routes/, middleware in src/middleware/.\n"
            f"{ts_note}"
        )

    # Generic Node.js with TypeScript
    if has_ts:
        return (
            "\n## Language: TypeScript\n"
            "This project uses TypeScript. Write ALL code as .ts/.tsx files using "
            "ES module syntax (import/export), NOT .js files with require()."
        )

    return ""


def _salvage_broken_tool_call(raw: str) -> Optional[Dict]:
    """
    When json.loads fails on a tool call candidate, try to extract
    the tool name and arguments using regex. This handles the common case
    where the model outputs write_file with markdown content that breaks JSON.
    """
    import re

    # Extract tool name
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
    if not name_match:
        return None
    name = name_match.group(1)

    if name == "write_file":
        # Extract path
        path_match = re.search(r'"path"\s*:\s*"([^"]+)"', raw)
        if not path_match:
            return None
        path = path_match.group(1)

        # Extract content: everything between "content": " and the last "
        content_match = re.search(r'"content"\s*:\s*"(.*)', raw, re.DOTALL)
        if not content_match:
            return None
        content_raw = content_match.group(1)
        # Remove trailing "} and clean up
        content_raw = re.sub(r'"\s*\}\s*\}\s*$', '', content_raw, flags=re.DOTALL)
        content_raw = re.sub(r'"\s*\}\s*$', '', content_raw, flags=re.DOTALL)
        # Unescape common JSON escapes
        file_content = (content_raw
                       .replace('\\n', '\n')
                       .replace('\\t', '\t')
                       .replace('\\"', '"')
                       .replace('\\\\', '\\'))

        return {
            "name": "write_file",
            "args": {"path": path, "content": file_content},
            "id": f"text_call_salvaged_{name}",
        }

    elif name == "task_done":
        summary_match = re.search(r'"summary"\s*:\s*"([^"]*)"', raw)
        summary = summary_match.group(1) if summary_match else "Task complete"
        files_created_match = re.search(r'"files_created"\s*:\s*"([^"]*)"', raw)
        files_created = files_created_match.group(1) if files_created_match else ""
        return {
            "name": "task_done",
            "args": {"summary": summary, "files_created": files_created},
            "id": f"text_call_salvaged_{name}",
        }

    return None


def _extract_write_from_prose(content: str) -> Optional[Dict]:
    """
    Last-resort extraction: when the model outputs something like:
      I'll create the file at docs/architecture/spec.md:
      ```markdown
      # Title
      content here...
      ```
    Extract it as a write_file call.
    """
    import re

    # Look for: path followed by a code block
    # Common patterns: "write to X:", "create X:", "file X:", etc.
    path_pattern = re.search(
        r'(?:write\s+(?:to\s+)?|create\s+|file\s+|save\s+(?:to\s+)?)'
        r'[`"]?([^\s`"]+\.(?:md|ts|tsx|js|jsx|json|css|yaml|yml))[`"]?',
        content, re.IGNORECASE,
    )
    if not path_pattern:
        return None

    path = path_pattern.group(1)

    # Extract the code block content
    code_match = re.search(r'```\w*\n(.*?)```', content, re.DOTALL)
    if not code_match:
        return None

    file_content = code_match.group(1)
    if len(file_content.strip()) < 50:  # Too short to be meaningful
        return None

    return {
        "name": "write_file",
        "args": {"path": path, "content": file_content},
        "id": "text_call_prose_extracted",
    }


def _auto_extract_and_write(
    content: str, subtask: str, tool_map: dict
) -> Optional[tuple]:
    """
    Last-resort auto-extraction: when the agent repeatedly outputs text
    instead of using write_file, try to extract a file path and content
    from the text and write it automatically.

    Returns (path, write_result) if successful, None otherwise.
    """
    import re

    # Strategy 1: Find explicit file path mentions + code blocks
    # Look for paths like "docs/architecture/spec.md", "src/models/User.ts", etc.
    path_patterns = re.findall(
        r'[`"\']*([a-zA-Z][\w/\\.-]+\.(?:md|ts|tsx|js|jsx|json|css|yaml|yml|mjs|mts))[`"\']*',
        content,
    )

    # Strategy 2: Infer path from the subtask if it mentions "write to X" or "write spec to X"
    if not path_patterns:
        task_path = re.search(
            r'(?:write\s+(?:to|spec\s+to|it\s+to))\s+[`"]?([^\s`"]+\.\w+)',
            subtask, re.IGNORECASE,
        )
        if task_path:
            path_patterns = [task_path.group(1)]

    if not path_patterns:
        return None

    # Extract the largest code block from the content
    code_blocks = re.findall(r'```\w*\n(.*?)```', content, re.DOTALL)

    # If no code blocks, try to extract everything after a heading or intro line
    if not code_blocks:
        # Look for substantial content (lines starting with #, -, or normal text)
        lines = content.strip().split('\n')
        # Skip the first few lines (usually "I'll create the file..." preamble)
        substantial = []
        started = False
        for line in lines:
            if not started and (line.startswith('#') or line.startswith('##')):
                started = True
            if started:
                substantial.append(line)
        if len(substantial) > 5:
            code_blocks = ['\n'.join(substantial)]

    if not code_blocks:
        return None

    # Use the largest code block
    file_content = max(code_blocks, key=len)
    if len(file_content.strip()) < 30:
        return None

    # Use the first file path that looks like a destination (not a source reference)
    target_path = path_patterns[0]

    # Execute write_file through the actual tool
    if "write_file" in tool_map:
        try:
            result = tool_map["write_file"].invoke({
                "path": target_path,
                "content": file_content,
            })
            return target_path, str(result)
        except Exception as e:
            return None

    return None


def _extract_text_tool_calls(content: str) -> List[Dict]:
    """
    Fallback parser: if the model outputs tool calls as JSON in its text
    response instead of using native function calling, try to extract them.

    Looks for patterns like:
      {"name": "write_file", "arguments": {"path": "...", "content": "..."}}
    or the ```json fenced variant.
    """
    if not content:
        return []

    import re
    candidates = []

    # Pattern 1: ```json ... ``` (multiline, handles nested content)
    for match in re.finditer(r"```json\s*(.*?)```", content, re.DOTALL):
        candidates.append(match.group(1).strip())

    # Pattern 2: bare JSON objects with "name" key — use a brace-counting
    # approach to handle nested objects (like arguments with nested dicts)
    i = 0
    while i < len(content):
        # Look for {"name" pattern
        idx = content.find('"name"', i)
        if idx == -1:
            break
        # Walk back to find opening brace
        brace_start = content.rfind('{', max(0, idx - 20), idx)
        if brace_start == -1:
            i = idx + 1
            continue
        # Brace-count forward to find matching close
        depth = 0
        j = brace_start
        while j < len(content):
            if content[j] == '{':
                depth += 1
            elif content[j] == '}':
                depth -= 1
                if depth == 0:
                    candidates.append(content[brace_start:j + 1])
                    break
            j += 1
        i = j + 1 if j < len(content) else idx + 1

    tool_calls = []
    seen_names = set()
    for raw in candidates:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "name" in parsed:
                name = parsed["name"]
                if name in seen_names:
                    continue
                seen_names.add(name)
                args = parsed.get("arguments", parsed.get("args", parsed.get("parameters", {})))
                tool_calls.append({
                    "name": name,
                    "args": args if isinstance(args, dict) else {},
                    "id": f"text_call_{len(tool_calls)}",
                })
        except (json.JSONDecodeError, TypeError):
            # JSON parsing failed — try regex extraction for write_file calls
            # This handles cases where content has unescaped newlines/special chars
            tc = _salvage_broken_tool_call(raw)
            if tc and tc["name"] not in seen_names:
                seen_names.add(tc["name"])
                tool_calls.append(tc)
            continue

    # Last resort: if no tool calls found, try to extract write_file from entire content
    # Models sometimes output: I'll write to path X:\n```\ncontent\n```
    if not tool_calls:
        tc = _extract_write_from_prose(content)
        if tc:
            tool_calls.append(tc)

    return tool_calls


# ── Error classification ──────────────────────────────────────────

_ERROR_PATTERNS = {
    "ImportError": ["importerror", "modulenotfounderror", "no module named", "cannot import"],
    "TypeError": ["typeerror", "is not callable", "expected str", "expected int", "argument"],
    "SyntaxError": ["syntaxerror", "unexpected token", "invalid syntax"],
    "FileNotFoundError": ["filenotfounderror", "no such file", "enoent"],
    "ConnectionError": ["connectionerror", "connection refused", "econnrefused"],
    "PermissionError": ["permissionerror", "permission denied", "eacces"],
    "BuildError": ["build failed", "compilation error", "tsc", "type error"],
    "TestFailure": ["failed", "assert", "expected", "pytest", "test_"],
    "DependencyError": ["npm err", "pip err", "package not found", "peer dep"],
}


def _classify_error(messages: list) -> tuple[str, str]:
    """
    Scan recent messages for error patterns. Returns (error_type, file_path).
    Looks at ToolMessage and HumanMessage content for known error signatures.
    """
    import re

    error_type = "unknown"
    error_file = ""

    # Scan the last 10 messages for errors
    for msg in reversed(messages[-10:]):
        content = getattr(msg, "content", "") or ""
        if not content:
            continue
        content_lower = content.lower()

        # Classify error type
        if error_type == "unknown":
            for etype, patterns in _ERROR_PATTERNS.items():
                if any(p in content_lower for p in patterns):
                    error_type = etype
                    break

        # Try to extract file path from error messages
        if not error_file:
            # Common patterns: "File '/path/to/file.py', line 42"
            # or "at /path/to/file.ts:42:10"
            file_match = re.search(
                r"""(?:File ['"]([^'"]+)['"]\s*,\s*line)|"""
                r"""(?:at\s+([^\s:]+\.\w+):\d+)""",
                content,
            )
            if file_match:
                error_file = file_match.group(1) or file_match.group(2) or ""

        if error_type != "unknown" and error_file:
            break

    return error_type, error_file


# ── Context window management ─────────────────────────────────────

# Compress message history every N iterations to prevent context overflow.
# Keeps the system prompt, task prompt, and recent messages in full.
# Older tool results get replaced with short summaries.
_COMPRESS_EVERY = 8       # Check every N iterations
_KEEP_RECENT = 6          # Keep the last N messages in full (3 LLM turns + 3 tool results)
_SUMMARY_MAX_CHARS = 120  # Max chars for compressed tool result summaries


def _compress_messages(messages: list, keep_first: int = 2) -> list:
    """
    Compress old messages to save context window space.

    Strategy:
    - Always keep the first `keep_first` messages (system + task prompt)
    - Always keep the last `_KEEP_RECENT` messages in full
    - For everything in between: replace ToolMessage content with a short summary
      and replace verbose AIMessage content with a truncated version
    """
    total = len(messages)
    if total <= keep_first + _KEEP_RECENT:
        return messages  # Nothing to compress

    compressed = messages[:keep_first]  # system + task

    middle_end = total - _KEEP_RECENT
    for msg in messages[keep_first:middle_end]:
        if isinstance(msg, ToolMessage):
            content = msg.content or ""
            # Summarize: first line + truncate
            first_line = content.split("\n")[0][:_SUMMARY_MAX_CHARS]
            compressed.append(ToolMessage(
                content=f"[compressed] {first_line}",
                tool_call_id=msg.tool_call_id,
            ))
        elif isinstance(msg, AIMessage):
            # Keep tool_calls metadata but truncate text content
            new_msg = AIMessage(
                content=(msg.content or "")[:200],
                tool_calls=msg.tool_calls if hasattr(msg, "tool_calls") else [],
            )
            compressed.append(new_msg)
        else:
            # HumanMessage or other — keep but truncate
            compressed.append(msg)

    # Keep recent messages in full
    compressed.extend(messages[middle_end:])

    old_chars = sum(len(getattr(m, "content", "") or "") for m in messages)
    new_chars = sum(len(getattr(m, "content", "") or "") for m in compressed)
    if old_chars > new_chars:
        rprint(f"  [dim]Context compressed: {old_chars:,} -> {new_chars:,} chars "
               f"({len(messages)} -> {len(compressed)} messages)[/dim]")

    return compressed


# ── Build verification ────────────────────────────────────────────

def _detect_build_commands() -> list[str]:
    """
    Detect what build/check commands to run based on the project type.
    Returns a list of commands to try. Empty list if nothing to check.
    """
    import subprocess, platform

    commands = []
    root = Path(PROJECT_ROOT)

    # Node.js / TypeScript project
    pkg_json = root / "package.json"
    if pkg_json.exists():
        try:
            pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
            scripts = pkg.get("scripts", {})

            # Prefer type-check or build script
            if "typecheck" in scripts:
                commands.append("npm run typecheck")
            elif "type-check" in scripts:
                commands.append("npm run type-check")
            elif "build" in scripts:
                commands.append("npm run build")
            elif "lint" in scripts:
                commands.append("npm run lint")

            # Also check if tsc is available for TS projects
            tsconfig = root / "tsconfig.json"
            if tsconfig.exists() and not commands:
                commands.append("npx tsc --noEmit")

        except (json.JSONDecodeError, OSError):
            pass

    # Python project
    if (root / "pyproject.toml").exists() or (root / "setup.py").exists():
        if (root / "tests").exists():
            commands.append("python -m pytest tests/ --tb=short -q")

    # If there's a requirements.txt but no package manager config, just check syntax
    if not commands and (root / "requirements.txt").exists():
        # Check if any .py files were recently modified by looking at the project
        if any((root / "app").glob("**/*.py")) if (root / "app").exists() else False:
            commands.append("python -m py_compile app/main.py")

    return commands


def _run_build_check(commands: list[str]) -> tuple[bool, str]:
    """
    Run build verification commands. Returns (success, error_output).
    """
    import subprocess, platform

    for cmd in commands:
        try:
            use_shell = platform.system() == "Windows"
            result = subprocess.run(
                cmd,
                shell=use_shell,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                error = result.stderr.strip() or result.stdout.strip()
                return False, f"Build check failed: `{cmd}`\n\n{error[:2000]}"
        except subprocess.TimeoutExpired:
            return False, f"Build check timed out: `{cmd}`"
        except Exception as e:
            return False, f"Build check error: `{cmd}` — {e}"

    return True, ""


# ── The main ReAct loop ───────────────────────────────────────────

def run_agent_loop(
    state: AgentState,
    agent_name: str,
    system_prompt: str,
    tools: List[BaseTool],
    memory=None,
    state_store=None,
) -> dict:
    """
    The autonomous agent execution loop (ReAct pattern).

    Instead of generating text and dumping it to a file, the agent:
    1. Reads the project to understand what exists
    2. Writes code to the REAL project directory
    3. Runs code to verify it works
    4. Fixes errors if any
    5. Calls task_done when finished

    This is what makes the system autonomous — agents interact with the
    real world through tools, just like a human developer.
    """
    global _last_done_result
    _last_done_result = {}

    llm = get_llm("specialist")

    # ── Generate unique task ID for this agent run ────────────────
    task_id = str(uuid.uuid4())[:8]
    run_id = get_run_id() or "no_run"

    # Build tool set: agent's tools + the task_done sentinel
    all_tools = list(tools) + [task_done]
    tool_map = {t.name: t for t in all_tools}

    # Bind tools to LLM for native function calling
    llm_with_tools = llm.bind_tools(all_tools)

    # ── Extract subtask + design doc from orchestrator message ─────
    last_msg = state["messages"][-1] if state["messages"] else {}
    design_doc = ""
    if isinstance(last_msg.get("content"), dict):
        subtask = last_msg["content"].get("subtask", state["task"])
        design_doc = last_msg["content"].get("design_doc", "")
    else:
        subtask = state.get("task", "No task provided")

    # ── Initialize stability systems ──────────────────────────────
    logger = AgentLogger(run_id, agent_name, task_id)
    cycle_detector = CycleDetector()
    agent_start_time = time.time()

    # Set agent context for tool hooks (write scoping, validation)
    from tools.hooks import set_current_agent
    set_current_agent(agent_name)

    # Gather files produced by previous agents in this plan
    prev_files = state.get("files_changed", [])

    # ── Gather RAG context ────────────────────────────────────────
    rag_context = ""
    if memory:
        try:
            from config import RAG_CODEBASE_K, RAG_MISTAKES_K
            code_ctx = memory.retrieve_context(subtask, k=RAG_CODEBASE_K)
            mistake_ctx = memory.retrieve_mistakes(subtask, k=RAG_MISTAKES_K)
            parts = [p for p in [code_ctx, mistake_ctx] if p]
            rag_context = "\n\n".join(parts)
        except Exception as e:
            rprint(f"[dim]RAG context failed: {e}[/dim]")

    # ── Build the task prompt ─────────────────────────────────────
    prompt_parts = [f"## Your Task\n{subtask}"]

    if design_doc:
        prompt_parts.append(
            f"\n## Technical Design Document (from Architect)\n"
            f"Follow this spec closely — it defines the contracts, file paths, "
            f"and data models all agents share.\n\n{design_doc}"
        )

    prompt_parts.append(f"\n## Project Directory\n{PROJECT_ROOT}")

    if prev_files:
        prompt_parts.append(
            f"\n## Files From Previous Agents (read these first!)\n"
            + "\n".join(f"- {f}" for f in prev_files)
        )

    if rag_context:
        prompt_parts.append(f"\n## Context From Memory\n{rag_context}")

    # Detect project type and inject framework-specific guidance
    framework_hint = _detect_project_framework()
    if framework_hint:
        prompt_parts.append(framework_hint)

    prompt_parts.append(
        "\n## How to Work\n"
        "1. Use list_directory and read_file to understand the project first\n"
        "2. Write code to the project directory using write_file\n"
        "3. Use run_command to test your code\n"
        "4. If errors occur, read them, fix the code, and retry\n"
        "5. When everything works, call task_done with a summary\n\n"
        "IMPORTANT: Write files to the ACTUAL project, not to an output/ folder.\n"
        "IMPORTANT: Always call task_done when finished.\n"
        "IMPORTANT: Do NOT overwrite tsconfig.json, next.config.ts, or "
        "other root config files unless your task specifically requires it."
    )

    task_prompt = "\n".join(prompt_parts)

    rprint(Panel(
        f"[bold]Task:[/bold] {subtask[:300]}\n"
        f"[dim]run={run_id} task={task_id} timeout={AGENT_TIMEOUT_SECONDS}s[/dim]",
        title=f"[cyan]{agent_name.upper()} Agent[/cyan]",
        subtitle="Starting autonomous execution",
    ))

    # ── ReAct loop: Think → Act → Observe → repeat ───────────────
    from config import LLM_PROVIDER
    # Append /nothink for local models (qwen3) to disable expensive thinking mode.
    # Not needed for Claude — it uses tool calling natively.
    suffix = "\n\n/nothink" if LLM_PROVIDER == "ollama" else ""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=task_prompt + suffix),
    ]

    files_changed = []
    iteration = 0
    final_output = ""
    build_retries = 0
    consecutive_nudges = 0  # Track repeated text-only responses

    for iteration in range(MAX_TOOL_ITERATIONS):
        # ── Wall-clock timeout check ─────────────────────────────
        elapsed = time.time() - agent_start_time
        if elapsed > AGENT_TIMEOUT_SECONDS:
            rprint(f"  [red]TIMEOUT: {agent_name} exceeded {AGENT_TIMEOUT_SECONDS}s "
                   f"({elapsed:.0f}s elapsed) — force stopping[/red]")
            logger.log("timeout", {"elapsed_s": round(elapsed, 2), "iteration": iteration})
            final_output = (f"Agent {agent_name} timed out after {elapsed:.0f}s "
                          f"at iteration {iteration}")
            break

        # ── Compress context if getting long ──────────────────────
        if iteration > 0 and iteration % _COMPRESS_EVERY == 0:
            messages = _compress_messages(messages)

        # ── Think: ask LLM what to do next ────────────────────────
        try:
            response = _invoke_with_retry(llm_with_tools, messages)
        except Exception as e:
            rprint(f"  [red]LLM call failed at step {iteration + 1}: {e}[/red]")
            logger.log("llm_error", {"error": str(e)[:300], "iteration": iteration})
            final_output = f"Agent {agent_name} LLM error: {e}"
            break

        messages.append(response)

        # ── Check for tool calls ──────────────────────────────────
        tool_calls = response.tool_calls or []

        # Fallback: if model wrote tool calls as text instead of using
        # native function calling, try to parse them from content
        if not tool_calls and response.content:
            text_calls = _extract_text_tool_calls(response.content)
            if text_calls:
                tool_calls = text_calls
                rprint(f"  [dim]Step {iteration + 1}: parsed {len(text_calls)} "
                       f"tool call(s) from text[/dim]")

        # No tool calls at all — LLM gave a final text response
        if not tool_calls:
            # If agent hasn't called task_done yet and we have iterations left,
            # nudge it to use tools instead of dumping text
            if not _last_done_result and iteration < MAX_TOOL_ITERATIONS - 2:
                consecutive_nudges += 1

                # After 3+ consecutive nudges, try to auto-extract content
                # and write it to a file. The model clearly can't produce
                # tool calls for large content.
                if consecutive_nudges >= 3 and response.content:
                    auto_written = _auto_extract_and_write(
                        response.content, subtask, tool_map
                    )
                    if auto_written:
                        path, write_result = auto_written
                        rprint(f"  [dim]Step {iteration + 1}: auto-extracted "
                               f"content → {path}[/dim]")
                        messages.append(ToolMessage(
                            content=write_result,
                            tool_call_id=f"auto_write_{iteration}",
                        ))
                        if path not in files_changed:
                            files_changed.append(path)
                        consecutive_nudges = 0
                        continue

                rprint(f"  [dim]Step {iteration + 1}: text response without tool calls "
                       f"— nudging agent to use tools "
                       f"(nudge {consecutive_nudges})[/dim]")
                messages.append(HumanMessage(
                    content=(
                        "You gave a text response but did NOT call any tools. "
                        "Do NOT write code or file contents as text — use the write_file tool "
                        "to create files. Use tools to take actions, then call task_done "
                        "when finished. Continue working.\n\n"
                        "EXAMPLE of correct tool usage:\n"
                        "Call write_file with path=\"your/file/path.ts\" and "
                        "content=\"your file content here\""
                    )
                ))
                continue
            final_output = response.content or "No output produced"
            rprint(f"  [dim]Step {iteration + 1}: final response (no tool calls)[/dim]")
            break

        # ── Act + Observe: execute each tool call ─────────────────
        consecutive_nudges = 0  # Reset on successful tool call
        done_this_round = False
        build_failed = False

        for tc in tool_calls:
            tool_name = tc.get("name", "unknown")
            tool_args = tc.get("args", {})
            tool_id = tc.get("id", f"call_{iteration}_{tool_name}")

            rprint(f"  [dim]Step {iteration + 1}: "
                   f"{tool_name}({_summarize_args(tool_args)})[/dim]")

            # ── Cycle detection ──────────────────────────────────
            if tool_name != "task_done":
                cycle_msg, cycle_severity = cycle_detector.check(tool_name, tool_args)
                if cycle_msg:
                    color = "red" if cycle_severity == "stop" else "yellow"
                    rprint(f"  [{color}]{cycle_msg[:140]}[/{color}]")
                    logger.log("cycle_detected", {
                        "tool": tool_name, "iteration": iteration,
                        "severity": cycle_severity,
                        "consecutive": cycle_detector._consecutive_count,
                    })
                    messages.append(ToolMessage(
                        content=cycle_msg,
                        tool_call_id=tool_id,
                    ))
                    if cycle_severity == "stop":
                        rprint(f"  [red]Hard stop — terminating {agent_name} "
                               f"to prevent runaway loop[/red]")
                        final_output = (
                            f"Agent {agent_name} terminated: stuck in cycle "
                            f"on {tool_name}. Partial work may remain."
                        )
                        done_this_round = True
                        break
                    # On "warn", break out of the inner tool loop so the agent
                    # sees the warning and reconsiders on the next iteration.
                    break

            # Handle task_done — agent signals completion
            if tool_name == "task_done":
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                except Exception:
                    result = f"Done: {tool_args.get('summary', 'task complete')}"

                messages.append(ToolMessage(
                    content=str(result), tool_call_id=tool_id,
                ))
                final_output = _last_done_result.get("summary", str(result))
                files_changed.extend(
                    _last_done_result.get("files_created", [])
                    + _last_done_result.get("files_modified", [])
                )

                # ── File existence check ─────────────────────────
                # Verify files the agent claims to have created actually exist.
                # This catches the common hallucination where the model calls
                # task_done with files_created but never actually wrote them.
                claimed_files = _last_done_result.get("files_created", [])
                missing_files = []
                for f in claimed_files:
                    fp = Path(f)
                    if not fp.is_absolute():
                        fp = PROJECT_ROOT / f
                    if not fp.exists():
                        missing_files.append(f)

                if missing_files:
                    rprint(f"  [yellow]Files claimed but NOT on disk: "
                           f"{', '.join(missing_files)} — rejecting task_done[/yellow]")
                    _last_done_result = {}
                    messages.append(HumanMessage(
                        content=(
                            f"TASK NOT COMPLETE. You claimed to create these files "
                            f"but they do NOT exist on disk:\n"
                            + "\n".join(f"  - {f}" for f in missing_files)
                            + "\n\nYou must use the write_file tool to actually "
                            f"create them. task_done does NOT write files — "
                            f"it only marks the task as complete. "
                            f"Write the files first, THEN call task_done."
                        )
                    ))
                    build_failed = True  # Reuse flag to prevent done_this_round
                    break

                # ── Build verification ────────────────────────────
                # Skip for testing agent — it IS the build verifier
                # Skip build verify for doc-only agents (they don't write code)
                _skip_build = agent_name in ("testing", "architect", "uiux", "reviewer")
                if BUILD_VERIFY_ENABLED and not _skip_build and build_retries < BUILD_VERIFY_MAX_RETRIES:
                    build_cmds = _detect_build_commands()
                    if build_cmds:
                        rprint(f"  [dim]Running build check: {', '.join(build_cmds)}[/dim]")
                        build_ok, build_error = _run_build_check(build_cmds)

                        if not build_ok:
                            # ── Scope check: only retry if the error is in
                            # files THIS agent created/modified. Pre-existing
                            # errors should NOT drag the agent into fix mode.
                            error_in_my_files = False
                            if files_changed:
                                for f in files_changed:
                                    # Normalize: strip leading ./ and project root
                                    fname = f.replace("\\", "/").split("/")[-1]  # filename
                                    fpath = f.replace("\\", "/")
                                    if fname in build_error or fpath in build_error:
                                        error_in_my_files = True
                                        break

                            if not error_in_my_files:
                                rprint(f"  [dim]Build has pre-existing errors "
                                       f"(not in this agent's files) — skipping retry[/dim]")
                                # Don't retry — let the agent finish
                            else:
                                build_retries += 1
                                rprint(f"  [yellow]Build check FAILED "
                                       f"(attempt {build_retries}/{BUILD_VERIFY_MAX_RETRIES})"
                                       f" — error in agent's own files, re-entering loop[/yellow]")
                                _last_done_result = {}
                                messages.append(HumanMessage(
                                    content=(
                                        f"BUILD VERIFICATION FAILED. There are errors in files YOU wrote. "
                                        f"Fix ONLY your files and call task_done again.\n\n"
                                        f"{build_error}"
                                    )
                                ))
                                build_failed = True
                                break  # Exit tool_calls loop, but NOT the ReAct loop
                        else:
                            rprint(f"  [green]Build check passed[/green]")

                if not build_failed:
                    rprint(f"  [green]Agent signaled completion at step "
                           f"{iteration + 1}[/green]")
                    done_this_round = True
                break

            # Execute the tool
            tool_start = time.time()
            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tool_args)

                    # Track file-modifying operations (only successful ones)
                    if tool_name in ("write_file", "edit_file", "append_file"):
                        result_str = str(result)
                        was_blocked = (
                            result_str.startswith("WRITE BLOCKED:")
                            or result_str.startswith("VALIDATION ERROR:")
                            or result_str.startswith("VALIDATION WARNING:")
                            or result_str.startswith("Error:")
                        )
                        if not was_blocked:
                            path = tool_args.get("path", "")
                            if path and path not in files_changed:
                                files_changed.append(path)

                except Exception as e:
                    result = f"Tool execution error: {e}"
            else:
                result = (
                    f"Error: Unknown tool '{tool_name}'. "
                    f"Available tools: {', '.join(tool_map.keys())}"
                )
            tool_duration = time.time() - tool_start
            logger.log_tool_call(iteration, tool_name, tool_args, str(result), tool_duration)

            # Feed result back to LLM (truncated to save context)
            result_text = _truncate(str(result))
            messages.append(ToolMessage(
                content=result_text,
                tool_call_id=tool_id,
            ))

            # ── Write-blocked correction ─────────────────────────
            # When an agent is blocked from writing, inject a firm
            # reminder so it stops retrying and does its actual job.
            if str(result).startswith("WRITE BLOCKED:"):
                from definitions.loader import get_agent_config
                try:
                    cfg = get_agent_config(agent_name)
                    allowed = cfg.get("write_scopes", [])
                except Exception:
                    allowed = []
                correction = (
                    f"IMPORTANT: You are the {agent_name} agent. You can ONLY write to: "
                    f"{allowed if allowed else '(unrestricted)'}. "
                    f"Do NOT attempt to write outside your scope again. "
                )
                if agent_name == "testing":
                    correction += (
                        "Your job is to REPORT bugs, not fix code. "
                        "Write your findings to docs/test-report.md and call task_done."
                    )
                else:
                    correction += (
                        "If you need a file created outside your scope, mention it in "
                        "your task_done summary so the responsible agent can do it."
                    )
                messages.append(HumanMessage(content=correction))

        if done_this_round:
            break

    else:
        rprint(f"  [yellow]Max iterations ({MAX_TOOL_ITERATIONS}) reached — "
               f"forcing stop[/yellow]")
        final_output = final_output or "Max iterations reached without completion"

    # ── Post-execution: log to memory ─────────────────────────────
    success = bool(_last_done_result) or (iteration < MAX_TOOL_ITERATIONS - 1)

    if memory:
        try:
            memory.embed_task_result(
                task=subtask,
                agent=agent_name,
                result=final_output[:1000],
                success=success,
                artifacts=files_changed,
            )
            if not success:
                # Extract structured error info from the conversation
                error_type, error_file = _classify_error(messages)
                memory.embed_mistake(
                    task=subtask,
                    agent=agent_name,
                    error=final_output[:500],
                    severity=2 if iteration < MAX_TOOL_ITERATIONS - 1 else 3,
                    error_type=error_type,
                    file_path=error_file,
                )
            # Also log build verification failures as separate mistakes
            if build_retries > 0 and success:
                memory.embed_mistake(
                    task=subtask,
                    agent=agent_name,
                    error=f"Build failed {build_retries} time(s) before passing",
                    fix="Agent self-corrected after build verification feedback",
                    severity=1,
                    error_type="build_failure",
                )
        except Exception as e:
            rprint(f"[dim]Memory logging failed: {e}[/dim]")

    if state_store:
        try:
            state_store.log_agent_action(
                agent=agent_name,
                action=f"Completed: {subtask[:100]}",
                result=final_output[:200],
                success=success,
                metadata={
                    "files_changed": files_changed,
                    "iterations": iteration + 1,
                },
            )
        except Exception:
            pass

    # ── Log completion ──────────────────────────────────────────
    logger.log_completion(success, iteration + 1, files_changed, final_output)

    total_time = time.time() - agent_start_time
    rprint(Panel(
        f"[bold green]Done[/bold green] — {iteration + 1} steps, "
        f"{len(files_changed)} files touched, {total_time:.1f}s\n"
        f"[dim]run={run_id} task={task_id}[/dim]\n"
        f"[bold]Files:[/bold] {', '.join(files_changed) if files_changed else 'none'}\n"
        f"[bold]Summary:[/bold] {final_output[:300]}",
        title=f"[cyan]{agent_name.upper()} Agent — Complete[/cyan]",
        subtitle="Returning to orchestrator",
    ))

    # Always return to orchestrator — it manages the plan
    return {
        "output": final_output[:500],
        "messages": [{
            "role": agent_name,
            "content": {
                "summary": final_output[:500],
                "files_changed": files_changed,
                "iterations": iteration + 1,
                "success": success,
            },
        }],
        "next_agent": "orchestrator",
        "retry_count": 0,
        "files_changed": files_changed,
    }
