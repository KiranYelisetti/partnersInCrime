"""
Tool hooks — pre/post execution validation inspired by claw-code.

Two systems:
1. Agent Write Scoping: restrict which paths each agent can write to
2. Pre-Write Validation: catch bad patterns before content hits disk

Write scopes are loaded from definitions/agents.yaml (single source of truth).
"""
import re
import threading
from pathlib import Path
from typing import Optional

from config import PROJECT_ROOT


# ── Agent context (thread-local) ─────────────────────────────────
# Set by agent_base before the ReAct loop, read by tools during execution.

_agent_context = threading.local()


def set_current_agent(name: str):
    _agent_context.name = name


def get_current_agent() -> str:
    return getattr(_agent_context, "name", "")


# ── Agent Write Scopes ───────────────────────────────────────────
# Loaded from definitions/agents.yaml — single source of truth.
# Each agent can only write to specific path prefixes.
# Empty list = unrestricted (e.g. infra scaffolds everything).

def _load_write_scopes():
    """Load write scopes from YAML. Falls back to empty dict on error."""
    try:
        from definitions.loader import get_all_write_scopes
        return get_all_write_scopes()
    except Exception:
        return {}

AGENT_WRITE_SCOPES = _load_write_scopes()


def check_write_scope(agent_name: str, file_path: str) -> Optional[str]:
    """
    Check if an agent is allowed to write to a given path.
    Returns an error message if blocked, None if allowed.
    """
    scopes = AGENT_WRITE_SCOPES.get(agent_name)

    # Unknown agent or unrestricted (empty list) = allow
    if scopes is None or len(scopes) == 0:
        return None

    # Normalize path to be relative to PROJECT_ROOT
    p = Path(file_path)
    if p.is_absolute():
        try:
            rel = str(p.resolve().relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            return None  # Outside project root — _safe_path will catch this
    else:
        rel = str(p)

    rel = rel.replace("\\", "/")  # Normalize Windows paths

    # Separate exclusion patterns (prefixed with "!") from inclusions
    excludes = [s[1:] for s in scopes if s.startswith("!")]
    includes = [s for s in scopes if not s.startswith("!")]

    # Check exclusions first — if any exclusion matches, block the write
    for exc in excludes:
        if rel.startswith(exc) or rel == exc.rstrip("/"):
            allowed = ", ".join(includes)
            return (
                f"WRITE BLOCKED: Agent '{agent_name}' is excluded from writing to '{rel}'. "
                f"Your allowed paths: [{allowed}] (excludes: {excludes}). "
                f"This file belongs to another agent. Focus on YOUR task only."
            )

    # Check if any inclusion prefix matches
    for scope in includes:
        if rel.startswith(scope) or rel == scope.rstrip("/"):
            return None

    # Blocked
    allowed = ", ".join(scopes)
    return (
        f"WRITE BLOCKED: Agent '{agent_name}' is not allowed to write to '{rel}'. "
        f"Your allowed paths: [{allowed}]. "
        f"Write files only within your scope. If you need to create a file "
        f"elsewhere, note it in your task_done summary so the responsible agent can do it."
    )


# ── Pre-Write Validation Hooks ───────────────────────────────────
# Catch common bad patterns before content is written to disk.

def validate_content(file_path: str, content: str) -> Optional[str]:
    """
    Validate file content before writing. Returns error message if invalid, None if OK.
    """
    if not isinstance(content, str):
        return None

    p = Path(file_path)
    ext = p.suffix.lower()

    # ── TypeScript/JavaScript validation ──────────────────────────
    if ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".mts"):

        # Block require() in TypeScript files
        if ext in (".ts", ".tsx", ".mts") and re.search(r'\brequire\s*\(', content):
            return (
                f"VALIDATION ERROR: '{p.name}' contains require() but this is a TypeScript file. "
                f"Use ES module imports: import X from 'module' instead of const X = require('module'). "
                f"Fix the content and try again."
            )

        # Block Express patterns in Next.js projects
        # Check if this is a Next.js project (next.config exists)
        next_config = PROJECT_ROOT / "next.config.ts"
        next_config_js = PROJECT_ROOT / "next.config.js"
        next_config_mjs = PROJECT_ROOT / "next.config.mjs"
        is_nextjs = next_config.exists() or next_config_js.exists() or next_config_mjs.exists()

        if is_nextjs:
            if re.search(r'from\s+[\'"]express[\'"]', content) or re.search(r'require\s*\(\s*[\'"]express[\'"]', content):
                return (
                    f"VALIDATION ERROR: '{p.name}' imports Express but this is a Next.js project. "
                    f"Use Next.js API routes (NextRequest/NextResponse) instead of Express. "
                    f"See the project type instructions in your task prompt."
                )

    # ── Reject non-ASCII in code files (catches Chinese characters from qwen3) ──
    if ext in (".ts", ".tsx", ".js", ".jsx", ".py", ".mjs", ".mts"):
        # Allow non-ASCII in strings (between quotes) but flag it in code
        # Simple heuristic: check for CJK characters outside of string literals
        for i, line in enumerate(content.splitlines(), 1):
            # Strip string contents to avoid false positives
            stripped = re.sub(r'(["\'])(?:(?!\1).)*\1', '""', line)
            # Check for CJK Unicode ranges
            if re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', stripped):
                return (
                    f"VALIDATION ERROR: Line {i} of '{p.name}' contains Chinese/CJK characters "
                    f"in code (outside of strings): '{line.strip()[:80]}'. "
                    f"This usually means the model generated non-English code. Fix it and retry."
                )

    # ── Next.js 'use client' directive validation ─────────────────
    if ext in (".tsx", ".jsx") and is_nextjs:
        first_line = content.split("\n", 1)[0].strip()
        # Check if file uses client hooks but lacks proper directive
        uses_hooks = re.search(r'\b(useState|useEffect|useRef|useCallback|useMemo|useContext)\b', content)
        if uses_hooks:
            # The directive must be a quoted string: "use client" or 'use client'
            has_proper_directive = first_line in ('"use client";', "'use client';", '"use client"', "'use client'")
            if not has_proper_directive:
                bad_forms = []
                if first_line.startswith("/*") and "use client" in first_line:
                    bad_forms.append(f"found comment: '{first_line}'")
                elif first_line == "use client" or first_line == "use client;":
                    bad_forms.append(f"found unquoted: '{first_line}'")
                hint = f" ({bad_forms[0]})" if bad_forms else ""
                return (
                    f"VALIDATION ERROR: '{p.name}' uses React hooks ({uses_hooks.group(1)}) "
                    f"but is missing the \"use client\" directive on line 1{hint}. "
                    f"The FIRST LINE must be exactly: \"use client\"; (with double quotes). "
                    f"Fix the content so line 1 is '\"use client\";' and try again."
                )

    # ── Warn on suspiciously small overwrites ─────────────────────
    # Exempt ephemeral report files — these are rewritten every pipeline run.
    _OVERWRITE_EXEMPT = {"review-report.md", "test-report.md"}
    resolved = PROJECT_ROOT / p if not p.is_absolute() else p
    if resolved.exists() and p.name not in _OVERWRITE_EXEMPT:
        existing_size = resolved.stat().st_size
        new_size = len(content.encode("utf-8"))
        # If new content is less than 20% of existing and existing is substantial
        if existing_size > 500 and new_size < existing_size * 0.2:
            return (
                f"VALIDATION WARNING: Writing {new_size} bytes to '{p.name}' would replace "
                f"{existing_size} bytes (>80% reduction). This likely means you're overwriting "
                f"a substantial file with a stub. Read the existing file first, then modify it "
                f"with edit_file instead of overwriting."
            )

    return None
