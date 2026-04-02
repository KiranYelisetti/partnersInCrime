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
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage
from rich import print as rprint
from rich.panel import Panel

from config import get_llm, LLM_COOLDOWN, RAG_RESULTS_K, RAG_MISTAKES_K, PROJECT_ROOT
from state import AgentState
from agents.agent_base import _invoke_with_retry


SYSTEM_PROMPT = """You are the orchestrator of a multi-agent dev team called "Partners in Crime".

Your agents and their EXACT responsibilities:
- architect: Full-stack tech lead. Reads the project, designs the TECHNICAL SPEC (API contracts, data models, file structure, integration points). Writes a design doc that all other agents follow. USE THIS FIRST when 2+ agents are involved.
- infra: Project setup and infrastructure. Creates package.json, tsconfig.json, config files, environment setup, Docker, CI/CD, installs dependencies with npm. ALWAYS include infra right after architect for new projects.
- database: Database models, schemas, ORM setup, DB connection logic — works with whatever DB the project uses (MongoDB/Mongoose, PostgreSQL/Prisma, SQLAlchemy, etc.)
- backend: API endpoints, business logic, authentication, server code — works with any framework (Next.js API routes, FastAPI, Express, etc.)
- frontend: React/Vue/Svelte components, TypeScript, UI logic, client-side code, pages
- uiux: Design specs, layouts, color tokens, component planning — BEFORE frontend builds
- testing: Writing AND RUNNING tests for existing code. Uses whatever test framework the project needs (vitest, jest, pytest, etc.). Runs the tests and fixes failures until green.

IMPORTANT CONTEXT: Each agent is AUTONOMOUS. They have tools to:
- Read files in the project
- Write/edit files in the project
- Run shell commands (python, pytest, npm, npx, node, etc.)
- Install packages (npm install, pip install)
- They work on the REAL project directory, not a sandbox

So your subtask descriptions should tell each agent WHAT to build, not HOW to use tools.
They know how to use their tools. Focus on the business requirements and technical specs.

ROUTING RULES:
1. For a NEW project (empty or near-empty dir) → "infra" → "architect" → "database" → "backend" → "frontend" → "testing"
   - INFRA FIRST: scaffolds the project (create-next-app, create-vite, etc.) so the directory has real configs
   - ARCHITECT SECOND: reads the scaffolded project + any reference code, writes the technical spec
   - Then specialists build on the real, working project base
2. If a task needs 2+ specialist agents on an EXISTING project → start with "architect"
3. If a task involves UI design decisions → "architect" → "uiux" → "frontend" → "testing"
4. If a task needs data models + API → "architect" → "database" → "backend" → "testing"
5. If a task is purely ONE agent's job (e.g., just fix a bug in one file) → skip architect
6. If a task is about deployment/docker only → "infra"
7. ALWAYS end with "testing" for any code changes — testing agent verifies the build works
8. If unclear or you need more info → "human"

RESPOND ONLY IN THIS JSON FORMAT:
{
  "plan": ["infra", "architect", "database", "backend", "frontend", "testing"],
  "plan_details": {
    "infra": "Scaffold Next.js project with create-next-app, install extra deps (mongoose, firebase-admin, etc.), create .env.example...",
    "architect": "Read the scaffolded project and v1 reference code. Design the full technical spec...",
    "database": "Follow the design doc. Create the database models/schemas...",
    "backend": "Follow the design doc. Create the API endpoints...",
    "frontend": "Follow the design doc. Create the React components and pages...",
    "testing": "Write tests and RUN them. Fix any failures. Verify npm run build passes."
  },
  "reasoning": "New project: infra scaffolds first, architect designs on real structure, then specialists build"
}

RULES:
- "plan" lists ALL agents needed in execution order
- For NEW projects, ALWAYS start with "infra" to scaffold, then "architect" to design
- For EXISTING projects needing multi-agent work, start with "architect"
- "plan_details" maps each agent name to its SPECIFIC subtask
- For architect: tell it WHAT feature to design (it decides the technical details)
- For specialists: tell them to FOLLOW the design doc AND what their piece is
- For single-agent tasks (e.g. "fix this one bug"), skip architect — just send directly
- If you need human input:
  {"plan": ["human"], "plan_details": {"human": "your question"}, "reasoning": "..."}
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

        if last_role in ("architect", "backend", "frontend", "database", "infra", "uiux", "testing"):
            next_step = current_step + 1

            # Collect what the last agent produced
            last_content = last_msg.get("content", {})
            last_files = []
            if isinstance(last_content, dict):
                last_files = last_content.get("files_changed", [])
            # Also accumulate from state
            all_files = state.get("files_changed", [])

            if next_step < len(plan):
                next_agent = plan[next_step]
                agent_subtask = plan_details.get(next_agent, state["task"])

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
                # Plan complete!
                rprint(Panel(
                    f"[bold green]All {len(plan)} plan steps complete![/bold green]\n"
                    f"[bold]Agents used:[/bold] {' -> '.join(plan)}\n"
                    f"[bold]Files created/modified:[/bold]\n"
                    + "\n".join(f"  - {f}" for f in all_files) if all_files else "  none",
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

    # Call LLM (/nothink only for local models to disable thinking mode)
    from config import LLM_PROVIDER
    suffix = "\n\n/nothink" if LLM_PROVIDER == "ollama" else ""
    response = _invoke_with_retry(llm, [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt + suffix),
    ])

    rprint(f"\n[dim]Orchestrator raw: {response.content[:400]}[/dim]\n")

    # Parse response
    try:
        clean = response.content.strip()
        # Strip thinking tags if model still produces them
        import re as _re
        clean = _re.sub(r'<think>.*?</think>', '', clean, flags=_re.DOTALL).strip()
        clean = clean.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        decision = json.loads(clean)

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
