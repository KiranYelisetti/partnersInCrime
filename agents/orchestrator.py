"""
Orchestrator — the parent agent that manages the team.

This is the "brain" of the system. It:
1. Reads the project structure to understand what exists
2. Creates a plan with per-agent subtasks
3. Dispatches agents one at a time (hub-and-spoke)
4. Passes context between agents (what files were created, etc.)
5. Ends when the plan is complete

Key difference from before: the orchestrator is now PROJECT-AWARE.
It reads the directory structure before planning so it can give agents
accurate context about what exists.
"""
import json
import os
import time
import uuid
from pathlib import Path
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage
from rich import print as rprint
from rich.panel import Panel

from config import get_llm, LLM_COOLDOWN, RAG_RESULTS_K, RAG_MISTAKES_K, PROJECT_ROOT, MAX_FIX_ROUNDS
from state import AgentState
from agents.agent_base import _invoke_with_retry, set_run_id, get_run_id


SYSTEM_PROMPT = """You are the orchestrator of a multi-agent dev team. You create a plan and dispatch agents.

AGENTS: infra, architect, database, backend, frontend, uiux, reviewer, testing
- infra: scaffolds project (create-next-app, etc.), installs deps
- architect: reads project + reference code, writes technical design doc AND api-contract.json
- database: creates data models/schemas following the design doc
- backend: creates API endpoints following the design doc and api-contract.json
- frontend: creates UI components/pages following the design doc and api-contract.json
- uiux: creates design specs (use BEFORE frontend if UI decisions needed)
- reviewer: checks frontend/backend alignment against api-contract.json, reports mismatches
- testing: runs build, reports bugs, verifies fixes

ROUTING:
- NEW project (empty dir) → infra → architect → database → backend → frontend → reviewer → testing
- EXISTING project, multi-agent → architect → specialists → reviewer → testing
- Single-agent fix → just that agent (no reviewer needed)
- Always put reviewer BEFORE testing (catches integration bugs cheaply)
- Always end with testing

RESPOND IN THIS EXACT JSON FORMAT ONLY:
{"plan":["infra","architect","database","backend","frontend","reviewer","testing"],"plan_details":{"infra":"what to do","architect":"what to do","database":"what to do","backend":"what to do","frontend":"what to do","reviewer":"review frontend/backend alignment against api-contract.json","testing":"what to do"},"reasoning":"why"}

RULES:
- plan_details: keep each subtask description to ONE sentence (max 50 words). Be concise.
- Each agent is autonomous with file read/write/run tools
- If you need human input: {"plan":["human"],"plan_details":{"human":"your question"},"reasoning":"need info"}
- reasoning: ONE sentence max
- Output ONLY valid JSON, nothing else
"""

_memory = None
_state_store = None


def set_dependencies(memory=None, state_store=None):
    global _memory, _state_store
    _memory = memory
    _state_store = state_store


def _read_design_docs() -> str:
    """
    Read all design docs from docs/architecture/.
    After the architect agent runs, it writes specs here.
    We inject this content into EVERY subsequent agent's context
    so they all follow the same contracts — no guessing.
    """
    arch_dir = PROJECT_ROOT / "docs" / "architecture"
    if not arch_dir.exists():
        return ""

    docs = []
    for f in sorted(arch_dir.iterdir()):
        if f.is_file() and f.suffix in (".md", ".json", ".txt", ".yaml", ".yml"):
            try:
                content = f.read_text(encoding="utf-8")
                if content.strip():
                    docs.append(f"=== {f.name} ===\n{content}")
            except Exception:
                continue

    if not docs:
        return ""

    combined = "\n\n".join(docs)
    # Cap at 4000 chars to leave room in context window
    if len(combined) > 4000:
        combined = combined[:4000] + "\n\n... (design doc truncated, read the full file with read_file tool)"

    return combined


def _get_project_structure(max_depth: int = 3) -> str:
    """
    Walk the project directory and return a tree string.
    This gives the orchestrator real knowledge of what exists.
    """
    root = Path(PROJECT_ROOT)
    skip_dirs = {
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        "chromadb_data", ".next", "dist", "build", ".mypy_cache",
        ".pytest_cache", "agents",  # skip the agent system itself
    }

    lines = [f"Project: {root}"]
    file_count = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip unwanted dirs
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]

        depth = Path(dirpath).relative_to(root).parts
        if len(depth) >= max_depth:
            dirnames.clear()
            continue

        indent = "  " * len(depth)
        dir_name = Path(dirpath).name
        if depth:
            lines.append(f"{indent}{dir_name}/")

        for f in sorted(filenames):
            if f.startswith(".") and f not in (".env", ".gitignore", ".dockerignore"):
                continue
            lines.append(f"{indent}  {f}")
            file_count += 1

        if file_count > 100:
            lines.append("  ... (truncated, too many files)")
            break

    return "\n".join(lines) if file_count > 0 else "Project directory is empty (new project)"


def _parse_review_report() -> dict:
    """
    Read docs/review-report.md written by the reviewer agent.
    Extract mismatches grouped by responsible agent (backend/frontend).
    Returns: {"has_mismatches": bool, "backend": [...], "frontend": [...], "raw": str}
    """
    import re

    report_path = PROJECT_ROOT / "docs" / "review-report.md"
    if not report_path.exists():
        return {"has_mismatches": False, "raw": "", "backend": [], "frontend": []}

    content = report_path.read_text(encoding="utf-8")

    if "status: pass" in content.lower():
        return {"has_mismatches": False, "raw": content, "backend": [], "frontend": []}

    errors_by_owner = {"backend": [], "frontend": []}

    # Split by "## Endpoint" blocks
    blocks = re.split(r'^## Endpoint', content, flags=re.MULTILINE)
    for block in blocks[1:]:  # skip header
        # Check for MISMATCH markers
        if "MISMATCH" not in block.upper():
            continue
        # Classify by which side has the mismatch
        if re.search(r'Backend.*MISMATCH', block, re.IGNORECASE):
            errors_by_owner["backend"].append("Endpoint" + block.strip())
        if re.search(r'Frontend.*MISMATCH', block, re.IGNORECASE):
            errors_by_owner["frontend"].append("Endpoint" + block.strip())

    # Fallback: if MISMATCH found but not classified, check for file paths
    if not any(errors_by_owner.values()) and "mismatch" in content.lower():
        for line in content.split("\n"):
            if "mismatch" in line.lower():
                if "backend" in line.lower() or "/api/" in line:
                    errors_by_owner["backend"].append(line.strip())
                elif "frontend" in line.lower() or "/component" in line:
                    errors_by_owner["frontend"].append(line.strip())

    has_mismatches = any(errors_by_owner.values())
    return {
        "has_mismatches": has_mismatches,
        "backend": errors_by_owner["backend"],
        "frontend": errors_by_owner["frontend"],
        "raw": content,
    }


def _parse_test_report() -> dict:
    """
    Read docs/test-report.md and extract errors grouped by responsible agent.
    Returns: {"backend": [errors...], "frontend": [errors...], "database": [errors...],
              "infra": [errors...], "build_passed": bool, "raw": str}

    Handles multiple report formats from different models:
      - "### Error N" blocks with "**Owner:** backend"
      - Numbered lists "1." with "**Owner**: Frontend" (colon outside bold)
      - Plain file-path mentions as fallback
    """
    import re

    report_path = PROJECT_ROOT / "docs" / "test-report.md"
    if not report_path.exists():
        return {"build_passed": False, "raw": "No test report found",
                "backend": [], "frontend": [], "database": [], "infra": []}

    content = report_path.read_text(encoding="utf-8")

    # Check if build passed
    if "build status: pass" in content.lower():
        return {"build_passed": True, "raw": content,
                "backend": [], "frontend": [], "database": [], "infra": []}

    errors_by_owner = {"backend": [], "frontend": [], "database": [], "infra": []}

    # ── Strategy 1: Find Owner tags in any format ────────────────
    # Matches: **Owner:** backend, **Owner**: Frontend, **owner**: BACKEND, etc.
    owner_pattern = re.compile(
        r'\*\*[Oo]wner\*?\*?[:\s]*[:\s]*(\w+)', re.IGNORECASE
    )

    # Split into error blocks — separated by "### Error", numbered "1.", "2.",
    # or "- **File**:" at the start of a line
    block_pattern = re.compile(
        r'^(?:###\s*[Ee]rror|\d+\.\s+\*\*[Ff]ile|- \*\*[Ff]ile)', re.MULTILINE
    )
    splits = block_pattern.split(content)
    boundaries = list(block_pattern.finditer(content))

    for i, boundary in enumerate(boundaries):
        block_start = boundary.start()
        block_end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(content)
        block_text = content[block_start:block_end].strip()

        # Find owner in this block
        owner_match = owner_pattern.search(block_text)
        if owner_match:
            owner = owner_match.group(1).lower()
            if owner in errors_by_owner:
                errors_by_owner[owner].append(block_text)

    # ── Strategy 2: Fallback — classify by file paths in error lines ─
    if not any(errors_by_owner.values()):
        for line in content.split("\n"):
            # Look for src/ file paths
            path_match = re.search(r'(src/[^\s:,\)]+\.\w+)', line)
            if not path_match:
                continue
            fpath = path_match.group(1)
            # Order matters: /app/api/ must be checked before /app/
            if "/app/api/" in fpath or "/lib/" in fpath or "/middleware/" in fpath or "/services/" in fpath:
                errors_by_owner["backend"].append(line.strip())
            elif "/models/" in fpath or "/db/" in fpath:
                errors_by_owner["database"].append(line.strip())
            elif "/components/" in fpath or "/hooks/" in fpath or "/pages/" in fpath:
                errors_by_owner["frontend"].append(line.strip())
            elif "/app/" in fpath:
                # /app/ pages (not /app/api/) are frontend
                errors_by_owner["frontend"].append(line.strip())

    # ── Strategy 3: Last resort — if build failed but no errors classified,
    # the report is unstructured. Assign everything to frontend (most common
    # build errors are missing 'use client', bad imports, etc.)
    if not any(errors_by_owner.values()) and "fail" in content.lower():
        errors_by_owner["frontend"].append(
            "Build failed but errors could not be classified. "
            "Read docs/test-report.md for details and fix all errors in your files."
        )

    return {
        "build_passed": False,
        "backend": errors_by_owner["backend"],
        "frontend": errors_by_owner["frontend"],
        "database": errors_by_owner["database"],
        "infra": errors_by_owner["infra"],
        "raw": content,
    }


def orchestrator_node(state: AgentState) -> dict:
    """
    The orchestrator node runs in three modes:

    1. FRESH TASK: No plan exists → read project, call LLM to create plan
    2. PLAN IN PROGRESS: Plan exists with steps remaining → dispatch next agent
    3. PLAN COMPLETE: All steps done → END
    """
    plan = state.get("agent_plan")
    plan_details = state.get("_plan_details")
    current_step = state.get("current_step", 0)

    # ── Mode 2 & 3: Existing plan — dispatch next or end ─────────
    if plan and len(plan) > 0 and plan_details:
        last_msg = state["messages"][-1] if state["messages"] else {}
        last_role = last_msg.get("role", "")

        if last_role in ("architect", "backend", "frontend", "database", "infra", "uiux", "reviewer", "testing"):
            next_step = current_step + 1

            # Collect what the last agent produced
            last_content = last_msg.get("content", {})
            last_files = []
            if isinstance(last_content, dict):
                last_files = last_content.get("files_changed", [])
            # Also accumulate from state
            all_files = list(dict.fromkeys(state.get("files_changed", [])))

            if next_step < len(plan):
                next_agent = plan[next_step]
                agent_subtask = plan_details.get(next_agent, state["task"])

                # ── Reviewer → Testing transition: check for mismatches ──
                # If reviewer just ran and found mismatches, inject a fix round
                # BEFORE testing so we don't waste an expensive build cycle.
                if last_role == "reviewer" and next_agent == "testing":
                    review = _parse_review_report()
                    if review["has_mismatches"]:
                        fix_plan = []
                        fix_details = {}
                        for agent_name in ["backend", "frontend"]:
                            agent_errors = review.get(agent_name, [])
                            if agent_errors:
                                fix_plan.append(agent_name)
                                error_text = "\n".join(agent_errors[:5])
                                fix_details[agent_name] = (
                                    f"REVIEW FIX: The integration reviewer found mismatches in YOUR code. "
                                    f"Read docs/review-report.md for details.\n\n"
                                    f"YOUR MISMATCHES:\n{error_text}\n\n"
                                    f"Read the review report, fix the mismatches to match "
                                    f"docs/architecture/api-contract.json, then call task_done."
                                )
                        # After fixes, continue with testing
                        fix_plan.append("testing")
                        fix_details["testing"] = plan_details.get("testing", state["task"])

                        rprint(Panel(
                            f"[bold yellow]REVIEWER FOUND MISMATCHES[/bold yellow]\n"
                            + "\n".join(f"  {a}: {len(review.get(a, []))} mismatch(es)" for a in ["backend", "frontend"])
                            + f"\n[bold]Fix plan:[/bold] {' -> '.join(fix_plan)}",
                            title="[yellow]Orchestrator — Review Fix[/yellow]",
                        ))

                        first_agent = fix_plan[0]
                        design_doc = _read_design_docs()

                        return {
                            "next_agent": first_agent,
                            "agent_plan": fix_plan,
                            "_plan_details": fix_details,
                            "current_step": 0,
                            "messages": [{"role": "orchestrator", "content": {
                                "subtask": fix_details[first_agent],
                                "reasoning": f"Reviewer found mismatches, fixing {first_agent} before testing",
                                "files_from_previous": all_files,
                                "design_doc": design_doc,
                            }}],
                            "needs_human": False,
                        }

                # Inject design docs so every agent follows the architect's spec
                design_doc = _read_design_docs()

                rprint(Panel(
                    f"[bold]Plan step {next_step + 1}/{len(plan)}:[/bold] {next_agent}\n"
                    f"[bold]Subtask:[/bold] {agent_subtask[:200]}\n"
                    f"[bold]Completed:[/bold] {' -> '.join(plan[:next_step + 1])}\n"
                    f"[bold]Remaining:[/bold] {' -> '.join(plan[next_step:])}\n"
                    f"[bold]Files so far:[/bold] {', '.join(all_files) if all_files else 'none'}\n"
                    f"[bold]Design doc:[/bold] {'injected (' + str(len(design_doc)) + ' chars)' if design_doc else 'none'}",
                    title="[magenta]Orchestrator — Continuing Plan[/magenta]",
                ))

                return {
                    "next_agent": next_agent,
                    "current_step": next_step,
                    "messages": [{"role": "orchestrator", "content": {
                        "subtask": agent_subtask,
                        "reasoning": f"Plan step {next_step + 1}/{len(plan)}: {next_agent}",
                        "files_from_previous": all_files,
                        "design_doc": design_doc,
                    }}],
                    "needs_human": False,
                }
            else:
                # ── Plan complete — check if we need fix loops ────
                fix_round = state.get("_fix_round", 0)

                # If testing just finished, check the report
                if last_role == "testing" and fix_round < MAX_FIX_ROUNDS:
                    report = _parse_test_report()

                    if report.get("build_passed"):
                        # Build is clean!
                        rprint(Panel(
                            f"[bold green]BUILD PASSED![/bold green] "
                            f"All agents done, build clean after {fix_round} fix round(s).\n"
                            f"[bold]Agents used:[/bold] {' -> '.join(plan)}\n"
                            f"[bold]Files:[/bold] {len(all_files)} files",
                            title="[green]Orchestrator — Build Clean[/green]",
                        ))
                        return {
                            "next_agent": "END",
                            "agent_plan": None,
                            "_plan_details": None,
                            "messages": [{"role": "orchestrator", "content": "Build passed — done"}],
                            "needs_human": False,
                        }

                    # Build failed — route fixes to responsible agents.
                    # Only include errors in files that agents actually
                    # created/modified during this pipeline run. Pre-existing
                    # broken files are not the agents' responsibility.
                    new_fix_round = fix_round + 1
                    fix_plan = []
                    fix_details = {}

                    # Normalize pipeline files for matching
                    pipeline_files = set()
                    for f in all_files:
                        normalized = f.replace("\\", "/").lstrip("./")
                        pipeline_files.add(normalized)
                        # Also add just the filename for fuzzy matching
                        pipeline_files.add(normalized.split("/")[-1])

                    for agent_name in ["database", "backend", "frontend", "infra"]:
                        agent_errors = report.get(agent_name, [])
                        # Filter: only keep errors that mention files this pipeline touched
                        relevant_errors = []
                        for err in agent_errors:
                            err_normalized = err.replace("\\", "/")
                            is_relevant = any(pf in err_normalized for pf in pipeline_files if len(pf) > 3)
                            if is_relevant:
                                relevant_errors.append(err)

                        if relevant_errors:
                            fix_plan.append(agent_name)
                            error_text = "\n".join(relevant_errors[:10])
                            fix_details[agent_name] = (
                                f"FIX ROUND {new_fix_round}: The tester found build errors in YOUR files. "
                                f"Read the test report at docs/test-report.md for full details.\n\n"
                                f"YOUR ERRORS TO FIX:\n{error_text}\n\n"
                                f"Steps:\n"
                                f"1. Read docs/test-report.md for the full error report\n"
                                f"2. Read the failing files\n"
                                f"3. Fix ONLY files you created — do NOT fix pre-existing files\n"
                                f"4. Call task_done when fixed"
                            )
                        elif agent_errors:
                            rprint(f"  [dim]{agent_name}: {len(agent_errors)} error(s) "
                                   f"in pre-existing files — skipped[/dim]")

                    # If no relevant errors (all pre-existing), skip fix loop
                    if not fix_plan:
                        rprint(Panel(
                            f"[bold green]BUILD ERRORS ARE ALL PRE-EXISTING[/bold green]\n"
                            f"All {sum(len(report.get(a, [])) for a in ['backend', 'frontend', 'database', 'infra'])} "
                            f"error(s) are in files that existed before this pipeline ran.\n"
                            f"The agents' code is correct — skipping fix loop.",
                            title="[green]Orchestrator — Skipping Fix Loop[/green]",
                        ))
                        return {
                            "next_agent": "END",
                            "agent_plan": None,
                            "_plan_details": None,
                            "messages": [{"role": "orchestrator", "content": "Build has only pre-existing errors — agents' work is done"}],
                            "needs_human": False,
                        }

                    # End fix round with testing (retest)
                    fix_plan.append("testing")
                    fix_details["testing"] = (
                        f"VERIFICATION ROUND {new_fix_round}: "
                        f"Developers have attempted fixes. Re-run npm run build and "
                        f"write a new test report to docs/test-report.md."
                    )

                    rprint(Panel(
                        f"[bold yellow]FIX ROUND {new_fix_round}/{MAX_FIX_ROUNDS}[/bold yellow]\n"
                        f"[bold]Bugs found by tester:[/bold]\n"
                        + "\n".join(f"  {a}: {len(report.get(a, []))} error(s)" for a in ["backend", "frontend", "database", "infra"])
                        + f"\n[bold]Fix plan:[/bold] {' -> '.join(fix_plan)}",
                        title="[yellow]Orchestrator — Integration Fix Loop[/yellow]",
                    ))

                    first_agent = fix_plan[0]
                    design_doc = _read_design_docs()

                    return {
                        "next_agent": first_agent,
                        "agent_plan": fix_plan,
                        "_plan_details": fix_details,
                        "current_step": 0,
                        "_fix_round": new_fix_round,
                        "_test_report": report.get("raw", ""),
                        "messages": [{"role": "orchestrator", "content": {
                            "subtask": fix_details[first_agent],
                            "reasoning": f"Fix round {new_fix_round}: routing bugs to {first_agent}",
                            "files_from_previous": all_files,
                            "design_doc": design_doc,
                        }}],
                        "needs_human": False,
                    }

                # No more fix rounds or testing didn't just run — done
                rprint(Panel(
                    f"[bold green]All plan steps complete![/bold green]\n"
                    f"[bold]Agents used:[/bold] {' -> '.join(plan)}\n"
                    f"[bold]Fix rounds:[/bold] {fix_round}\n"
                    f"[bold]Files:[/bold] {len(all_files)} total",
                    title="[green]Orchestrator — Plan Complete[/green]",
                ))
                return {
                    "next_agent": "END",
                    "agent_plan": None,
                    "_plan_details": None,
                    "messages": [{"role": "orchestrator", "content": "Plan complete"}],
                    "needs_human": False,
                }

    # ── Mode 1: Fresh task — read project + call LLM to plan ─────
    # Generate a unique run ID for this entire pipeline execution
    if not get_run_id():
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:6]
        set_run_id(run_id)
        rprint(f"[dim]Pipeline run ID: {run_id}[/dim]")

        # Clean up stale reports from previous pipeline runs so agents
        # don't read outdated data (especially review-report.md).
        for stale in ("docs/review-report.md", "docs/test-report.md"):
            stale_path = PROJECT_ROOT / stale
            if stale_path.exists():
                stale_path.unlink()
                rprint(f"  [dim]Cleaned stale {stale}[/dim]")

    llm = get_llm("orchestrator")

    # Read the actual project structure
    project_tree = _get_project_structure()

    # Gather RAG context
    rag_context = ""
    if _memory:
        try:
            task = state["task"]
            past = _memory.retrieve_past_results(task, k=RAG_RESULTS_K)
            mistakes = _memory.retrieve_mistakes(task, k=RAG_MISTAKES_K)
            parts = [p for p in [past, mistakes] if p]
            if parts:
                rag_context = "\n".join(parts)
        except Exception as e:
            rprint(f"[dim]Orchestrator RAG failed: {e}[/dim]")

    # Build prompt
    prompt_parts = [
        f"Task: {state['task']}",
        f"\n--- CURRENT PROJECT STRUCTURE ---\n{project_tree}\n--- END STRUCTURE ---",
    ]

    if state.get("messages"):
        recent = [m for m in state["messages"][-5:] if m["role"] != "orchestrator"]
        if recent:
            prompt_parts.append(f"\nPrevious agent results: {json.dumps(recent, default=str)[:500]}")

    if rag_context:
        prompt_parts.append(f"\n--- CONTEXT FROM MEMORY ---\n{rag_context}\n--- END CONTEXT ---")

    if state.get("error"):
        prompt_parts.append(f"\nPrevious attempt failed with: {state['error']}")

    prompt = "\n".join(prompt_parts)

    # Call LLM (thinking mode handled by reasoning=False in config.py)
    response = _invoke_with_retry(llm, [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    rprint(f"\n[dim]Orchestrator raw: {response.content[:400]}[/dim]\n")

    # Parse response — robust extraction that handles malformed JSON
    try:
        clean = response.content.strip()
        # Strip thinking tags if model still produces them
        import re as _re
        clean = _re.sub(r'<think>.*?</think>', '', clean, flags=_re.DOTALL).strip()
        clean = clean.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        # Try direct parse first
        decision = None
        try:
            decision = json.loads(clean)
        except json.JSONDecodeError:
            pass

        # Fallback: extract the first JSON object from the text
        if decision is None:
            brace_start = clean.find("{")
            if brace_start >= 0:
                depth = 0
                in_string = False
                escape_next = False
                for i, ch in enumerate(clean[brace_start:], brace_start):
                    if escape_next:
                        escape_next = False
                        continue
                    if ch == "\\":
                        escape_next = True
                        continue
                    if ch == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                decision = json.loads(clean[brace_start:i + 1])
                            except json.JSONDecodeError:
                                pass
                            break

        # Last resort: try to repair truncated JSON by closing brackets
        if decision is None and "{" in clean:
            fragment = clean[clean.find("{"):]
            # Close any open strings, arrays, objects
            fragment = fragment.rstrip(",\n\r\t ")
            open_braces = fragment.count("{") - fragment.count("}")
            open_brackets = fragment.count("[") - fragment.count("]")
            # Close open strings (crude)
            if fragment.count('"') % 2 != 0:
                fragment += '"'
            fragment += "]" * max(0, open_brackets)
            fragment += "}" * max(0, open_braces)
            try:
                decision = json.loads(fragment)
            except json.JSONDecodeError:
                pass

        if decision is None:
            raise json.JSONDecodeError("Could not extract valid JSON", clean, 0)

        plan = decision.get("plan", [])
        plan_details = decision.get("plan_details", {})
        reasoning = decision.get("reasoning", "")

        # Backward compat: if old format with "next_agent" / "subtask"
        if not plan and decision.get("next_agent"):
            agent = decision["next_agent"]
            subtask = decision.get("subtask", state["task"])
            plan = [agent]
            plan_details = {agent: subtask}

        # Validate
        if not plan:
            plan = ["human"]
            plan_details = {"human": "Could not determine which agent to use. Please clarify."}

        next_agent = plan[0]
        first_subtask = plan_details.get(next_agent, state["task"])

        rprint(Panel(
            f"[bold]Plan:[/bold] {' -> '.join(plan)}\n"
            + "\n".join(f"  [cyan]{a}:[/cyan] {plan_details.get(a, '?')[:150]}" for a in plan)
            + f"\n[bold]Reasoning:[/bold] {reasoning}",
            title="[magenta]Orchestrator Decision[/magenta]",
        ))

        # Log
        if _state_store:
            try:
                _state_store.log_agent_action(
                    agent="orchestrator",
                    action=f"Created plan: {' -> '.join(plan)}",
                    result=reasoning[:200],
                    metadata={"plan": plan, "plan_details": plan_details},
                )
            except Exception:
                pass

        # Include design doc if one already exists (e.g. re-running a plan)
        design_doc = _read_design_docs()

        return {
            "next_agent": next_agent,
            "messages": [{"role": "orchestrator", "content": {
                "subtask": first_subtask,
                "reasoning": reasoning,
                "files_from_previous": [],
                "design_doc": design_doc,
            }}],
            "needs_human": next_agent == "human",
            "context": rag_context if rag_context else None,
            "agent_plan": plan,
            "_plan_details": plan_details,
            "current_step": 0,
        }

    except (json.JSONDecodeError, Exception) as e:
        rprint(f"[red]Orchestrator parse error:[/red] {e}\n[dim]Raw: {response.content[:200]}[/dim]")
        return {
            "error": f"Orchestrator parse failed: {e}",
            "next_agent": "human",
            "needs_human": True,
            "messages": [{"role": "orchestrator", "content": f"Parse error: {e}"}],
        }
