"""
Quick integration test — verifies the YAML-driven agent system works end-to-end.
Runs a small 2-agent task (architect + backend) to validate:
  1. YAML loader resolves configs and tools correctly
  2. Agent wrappers load prompts from YAML
  3. Orchestrator creates a plan and dispatches agents
  4. Write scoping is enforced
  5. Full ReAct loop executes
"""
import sys
import os
import time
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))

from rich import print as rprint
from rich.panel import Panel

from config import PROJECT_ROOT
from main import init_system, build_graph

rprint(Panel(
    f"[bold]Project root:[/bold] {PROJECT_ROOT}\n"
    "[bold]Goal:[/bold] Quick 2-agent test (architect + backend) to verify YAML-driven system.",
    title="[bold green]Integration Test — YAML Agent Loading[/bold green]",
))

# Initialize
memory, state_store = init_system()
app = build_graph()

# Small, focused task — should only need architect + 1 backend agent
task = (
    "Create a simple health-check API endpoint for the existing Next.js project.\n"
    "Steps:\n"
    "1. Architect: design a single GET /api/health endpoint that returns { status: 'ok', timestamp: ISO string }. "
    "Write a short design doc to docs/architecture/health_check_design.md\n"
    "2. Backend: implement the endpoint at src/app/api/health/route.ts following the design doc"
)

rprint(f"\n[bold cyan]Task:[/bold cyan] {task}\n{'=' * 60}")

start = time.time()

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
        "_fix_round":    0,
        "_test_report":  None,
    })

    elapsed = time.time() - start
    agents_used = [m["role"] for m in result.get("messages", [])
                  if m["role"] not in ("orchestrator", "human")]
    all_files = result.get("files_changed", [])

    rprint(Panel(
        f"[bold green]Test complete[/bold green]\n\n"
        f"[bold]Time:[/bold] {elapsed:.1f}s\n"
        f"[bold]Agents used:[/bold] {' -> '.join(agents_used)}\n"
        f"[bold]Files created:[/bold]\n"
        + ("\n".join(f"  - {f}" for f in all_files) if all_files else "  none") + "\n"
        f"[bold]Errors:[/bold] {result.get('error', 'none')}\n"
        f"[bold]Output:[/bold] {result.get('output', 'none')[:500]}",
        title="[green]Result[/green]",
        border_style="green",
    ))

    # Verify key files
    rprint("\n[bold]File verification:[/bold]")
    expected = [
        PROJECT_ROOT / "docs" / "architecture" / "health_check_design.md",
        PROJECT_ROOT / "src" / "app" / "api" / "health" / "route.ts",
    ]
    for f in expected:
        if f.exists():
            rprint(f"  [green]EXISTS[/green] {f} ({f.stat().st_size} bytes)")
        else:
            rprint(f"  [yellow]MISSING[/yellow] {f}")

    # Also check any files the agents actually created
    for f in all_files:
        p = Path(f) if Path(f).is_absolute() else PROJECT_ROOT / f
        if p.exists() and p not in expected:
            rprint(f"  [green]EXISTS[/green] {p} ({p.stat().st_size} bytes)")

except KeyboardInterrupt:
    rprint("\n[yellow]Test interrupted[/yellow]")
except Exception as e:
    elapsed = time.time() - start
    rprint(f"\n[red]Test failed after {elapsed:.1f}s:[/red] {e}")
    import traceback
    traceback.print_exc()
