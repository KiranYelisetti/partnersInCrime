"""
Partners in Crime — Main entry point.

Builds the LangGraph agent graph, initializes memory (ChromaDB) and state store (Redis),
injects dependencies into all agents, and runs the interactive loop.

Graph flow (hub-and-spoke):
  User → Orchestrator → Agent (autonomous tool loop) → Orchestrator → Agent → ... → END

Each agent runs its own internal ReAct loop:
  Think → Tool Call → Observe → Think → Tool Call → ... → task_done
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langgraph.graph import StateGraph, END
from rich import print as rprint
from rich.panel import Panel
from rich.console import Console
from rich.table import Table

from state import AgentState
from config import PROJECT_ROOT, CHROMADB_PATH

# Import agent nodes
from orchestrator import orchestrator_node, set_dependencies as orch_set_deps
from agents.backend_agent import backend_node, set_dependencies as be_set_deps
from agents.frontend_agent import frontend_node, set_dependencies as fe_set_deps
from agents.database_agent import database_node, set_dependencies as db_set_deps
from agents.infra_agent import infra_node, set_dependencies as infra_set_deps
from agents.uiux_agent import uiux_node, set_dependencies as uiux_set_deps
from agents.testing_agent import testing_node, set_dependencies as test_set_deps
from agents.architect_agent import architect_node, set_dependencies as arch_set_deps
from agents.reviewer_agent import reviewer_node, set_dependencies as rev_set_deps

# Import memory layer
from memory.vector_store import AgentMemory
from memory.state_store import StateStore

console = Console()

VALID_AGENTS = {"architect", "backend", "frontend", "database", "infra", "uiux", "reviewer", "testing"}


# ── Initialize memory and state ──────────────────────────────────
def init_system():
    """Initialize AgentMemory and StateStore, inject into all agents."""
    rprint("[dim]Initializing memory layer (ChromaDB)...[/dim]")
    memory = AgentMemory(persist_dir=CHROMADB_PATH)
    stats = memory.get_stats()
    rprint(f"[dim]  ChromaDB ready — {stats['codebase_chunks']} code chunks, "
           f"{stats['task_results']} past results, {stats['mistakes']} mistakes[/dim]")

    rprint("[dim]Initializing state store...[/dim]")
    state_store = StateStore()
    rprint(f"[dim]  State store ready — mode: {state_store.mode}[/dim]")

    for setter in [orch_set_deps, arch_set_deps, be_set_deps, fe_set_deps,
                   db_set_deps, infra_set_deps, uiux_set_deps, rev_set_deps, test_set_deps]:
        setter(memory=memory, state_store=state_store)

    return memory, state_store


# ── Graph routing ─────────────────────────────────────────────────
def route_from_orchestrator(state: AgentState) -> str:
    if state.get("needs_human"):
        return "human_node"
    next_a = state.get("next_agent", "END")
    if next_a in VALID_AGENTS:
        return next_a
    return "END"


def human_node(state: AgentState) -> dict:
    """Human-in-the-loop: asks the user for input when orchestrator needs help."""
    last = state["messages"][-1]["content"] if state["messages"] else "Need input"
    question = last.get("question", last.get("subtask", "Need your input")) \
               if isinstance(last, dict) else str(last)
    rprint(Panel(
        question,
        title="[bold yellow]Orchestrator needs your input[/bold yellow]",
        border_style="yellow",
    ))
    answer = input("\n  Your answer -> ")
    return {
        "task": state["task"] + f"\n[Human clarified: {answer}]",
        "needs_human": False,
        "messages": [{"role": "human", "content": answer}],
    }


# ── Build graph ───────────────────────────────────────────────────
def build_graph():
    """
    Build the LangGraph with a hub-and-spoke pattern.

    Orchestrator is the HUB — all agents are spokes.
    Every agent returns to orchestrator after finishing.
    Orchestrator manages the plan and dispatches next agent or ends.
    """
    g = StateGraph(AgentState)

    # Register all nodes
    nodes = [
        ("orchestrator", orchestrator_node),
        ("architect",    architect_node),
        ("backend",      backend_node),
        ("frontend",     frontend_node),
        ("database",     database_node),
        ("infra",        infra_node),
        ("uiux",         uiux_node),
        ("reviewer",     reviewer_node),
        ("testing",      testing_node),
        ("human_node",   human_node),
    ]
    for name, fn in nodes:
        g.add_node(name, fn)

    # Entry point
    g.set_entry_point("orchestrator")

    # Orchestrator → any agent, human, or END
    g.add_conditional_edges("orchestrator", route_from_orchestrator, {
        "architect":  "architect",
        "backend":    "backend",
        "frontend":   "frontend",
        "database":   "database",
        "infra":      "infra",
        "uiux":       "uiux",
        "reviewer":   "reviewer",
        "testing":    "testing",
        "human_node": "human_node",
        "END":         END,
    })

    # EVERY specialist agent → back to orchestrator (hub-and-spoke)
    for agent in VALID_AGENTS:
        g.add_edge(agent, "orchestrator")

    # Human → back to orchestrator
    g.add_edge("human_node", "orchestrator")

    return g.compile()


# ── Main ──────────────────────────────────────────────────────────
def print_banner():
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column(style="white")
    table.add_row("Architect","Full-stack tech lead — designs contracts before anyone codes")
    table.add_row("Backend",  "API routes, business logic, auth — autonomous with tools")
    table.add_row("Frontend", "React, TypeScript, UI logic — autonomous with tools")
    table.add_row("Database", "Schemas, models, ORM setup — autonomous with tools")
    table.add_row("UI/UX",    "Design specs, layouts, tokens — autonomous with tools")
    table.add_row("Infra",    "Project setup, configs, Docker, CI/CD — autonomous with tools")
    table.add_row("Reviewer", "Integration review — catches frontend/backend mismatches")
    table.add_row("Testing",  "Writes & runs tests, fixes failures — autonomous with tools")

    console.print(Panel(
        table,
        title="[bold green]Partners in Crime — Autonomous Agent System[/bold green]",
        subtitle="[dim]All 8 agents online — architect designs, specialists build, reviewer validates[/dim]",
        border_style="green",
    ))


def main():
    print_banner()

    # Initialize system
    memory, state_store = init_system()

    # Embed codebase on first run
    if memory.get_stats()["codebase_chunks"] == 0:
        rprint("\n[yellow]First run detected — embedding codebase for RAG context...[/yellow]")
        try:
            stats = memory.embed_codebase(str(PROJECT_ROOT))
            rprint(f"[green]  Embedded {stats['files_processed']} files "
                   f"({stats['chunks_added']} chunks)[/green]")
        except Exception as e:
            rprint(f"[red]  Codebase embedding failed: {e}[/red]")
            rprint("[dim]  System will work without codebase context[/dim]")

    # Build graph
    app = build_graph()

    # Interactive loop
    rprint("\n[bold]Enter a task (or 'quit' to exit, 'status' for memory stats):[/bold]")

    while True:
        try:
            task = input("\n  Task -> ").strip()
        except (KeyboardInterrupt, EOFError):
            rprint("\n[dim]Shutting down...[/dim]")
            break

        if not task:
            continue
        if task.lower() in ("quit", "exit", "q"):
            rprint("[dim]Goodbye![/dim]")
            break
        if task.lower() == "status":
            _print_status(memory, state_store)
            continue
        if task.lower() == "reindex":
            rprint("[yellow]Re-embedding codebase...[/yellow]")
            try:
                stats = memory.embed_codebase(str(PROJECT_ROOT), force=True)
                rprint(f"[green]  {stats['files_processed']} files, "
                       f"{stats['chunks_added']} chunks[/green]")
            except Exception as e:
                rprint(f"[red]  Reindex failed: {e}[/red]")
            continue

        # Run the graph
        rprint(f"\n[bold cyan]Starting task:[/bold cyan] {task}\n{'─' * 60}")
        try:
            result = app.invoke({
                "task":          task,
                "messages":      [],
                "retry_count":   0,
                "needs_human":   False,
                "next_agent":    None,
                "output":        None,
                "error":         None,
                "context":       None,
                "agent_plan":    None,
                "current_step":  0,
                "_plan_details": None,
                "files_changed": [],
            })

            # Show result summary
            agents_used = [m["role"] for m in result.get("messages", [])
                          if m["role"] not in ("orchestrator", "human")]
            all_files = result.get("files_changed", [])

            summary_parts = [
                f"[bold green]Task complete[/bold green]\n",
                f"[bold]Agents used:[/bold] {' -> '.join(agents_used) if agents_used else 'none'}",
            ]
            if all_files:
                summary_parts.append(f"[bold]Files created/modified:[/bold]")
                for f in all_files:
                    summary_parts.append(f"  - {f}")
            if result.get("error"):
                summary_parts.append(f"[bold]Errors:[/bold] {result['error']}")

            rprint(Panel(
                "\n".join(summary_parts),
                title="[green]Result[/green]",
                border_style="green",
            ))

        except KeyboardInterrupt:
            rprint("\n[yellow]Task interrupted[/yellow]")
        except Exception as e:
            rprint(f"\n[red]Task failed:[/red] {e}")
            import traceback
            traceback.print_exc()


def _print_status(memory, state_store):
    """Print system status."""
    mem_stats = memory.get_stats()
    store_status = state_store.get_status()

    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Codebase chunks", str(mem_stats["codebase_chunks"]))
    table.add_row("Task results", str(mem_stats["task_results"]))
    table.add_row("Logged mistakes", str(mem_stats["mistakes"]))
    table.add_row("State store mode", store_status["mode"])
    table.add_row("Task queue length", str(store_status["queue_length"]))
    table.add_row("Action log entries", str(store_status["action_log_entries"]))

    console.print(table)


if __name__ == "__main__":
    main()
