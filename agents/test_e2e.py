"""
End-to-end test — runs a simple task through the full pipeline.
Tests: Orchestrator → Architect → Database → Backend → Testing
"""
import sys
import os
import time
from pathlib import Path

# Fix Windows Unicode output
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
sys.stderr.reconfigure(encoding="utf-8") if hasattr(sys.stderr, "reconfigure") else None

sys.path.insert(0, str(Path(__file__).parent))

from rich import print as rprint
from rich.panel import Panel

from config import PROJECT_ROOT
from main import init_system, build_graph

# Clean up any previous test output
test_dirs = [
    PROJECT_ROOT / "app",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "docs",
]
for d in test_dirs:
    if d.exists():
        import shutil
        shutil.rmtree(d)
        rprint(f"[dim]Cleaned up {d}[/dim]")


rprint(Panel(
    "Running a simple task through the full pipeline:\n"
    "Orchestrator → Architect → Database → Backend → Testing\n\n"
    "Task: 'Create a simple health check endpoint that returns the app version'",
    title="[bold green]E2E Test[/bold green]",
))

# Initialize
memory, state_store = init_system()

# Build graph
app = build_graph()

# Run
start = time.time()
task = "Create a simple health check endpoint at GET /health that returns {\"status\": \"ok\", \"version\": \"1.0.0\"}. Put the FastAPI app in app/main.py."

rprint(f"\n[bold cyan]Task:[/bold cyan] {task}\n{'─' * 60}")

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

    elapsed = time.time() - start
    agents_used = [m["role"] for m in result.get("messages", [])
                  if m["role"] not in ("orchestrator", "human")]
    all_files = result.get("files_changed", [])

    rprint(Panel(
        f"[bold green]E2E Test Complete[/bold green]\n\n"
        f"[bold]Time:[/bold] {elapsed:.1f}s\n"
        f"[bold]Agents used:[/bold] {' -> '.join(agents_used)}\n"
        f"[bold]Files created:[/bold]\n"
        + "\n".join(f"  - {f}" for f in all_files) + "\n"
        f"[bold]Errors:[/bold] {result.get('error', 'none')}\n"
        f"[bold]Final output:[/bold] {result.get('output', 'none')[:300]}",
        title="[green]Result[/green]",
        border_style="green",
    ))

    # Verify files actually exist
    rprint("\n[bold]Verification:[/bold]")
    for f in all_files:
        p = Path(f)
        if not p.is_absolute():
            p = PROJECT_ROOT / f
        if p.exists():
            size = p.stat().st_size
            rprint(f"  [green]EXISTS[/green] {p} ({size} bytes)")
        else:
            rprint(f"  [red]MISSING[/red] {p}")

except KeyboardInterrupt:
    rprint("\n[yellow]Test interrupted[/yellow]")
except Exception as e:
    rprint(f"\n[red]Test failed:[/red] {e}")
    import traceback
    traceback.print_exc()
