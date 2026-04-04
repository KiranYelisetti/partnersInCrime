"""
Test: Autonomous fix loop — verifies the system can:
  1. Detect build errors
  2. Route them to the right developer agent
  3. Have that agent fix the code
  4. Retest and pass

Uses the existing project which has known 'use client' build errors.
The orchestrator should: testing reports → frontend fixes → testing verifies.
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
    "[bold]Goal:[/bold] Test autonomous build-fix loop.\n"
    "The project has known 'use client' errors. System should:\n"
    "  1. Testing agent reports build errors\n"
    "  2. Frontend agent fixes them\n"
    "  3. Testing agent verifies the build passes",
    title="[bold green]Autonomous Fix Loop Test[/bold green]",
))

memory, state_store = init_system()
app = build_graph()

task = (
    "Fix all build errors in the project and make npm run build pass.\n"
    "Steps:\n"
    "1. Testing: run npm run build, report all errors to docs/test-report.md with file, error, owner, fix suggestion\n"
    "2. The responsible agents fix their errors\n"
    "3. Testing: verify the build passes"
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
        f"[bold]Time:[/bold] {elapsed:.1f}s ({elapsed/60:.1f} min)\n"
        f"[bold]Agents used:[/bold] {' -> '.join(agents_used)}\n"
        f"[bold]Files changed:[/bold] {len(all_files)}\n"
        + ("\n".join(f"  - {f}" for f in all_files) if all_files else "  none") + "\n"
        f"\n[bold]Errors:[/bold] {result.get('error', 'none')}\n"
        f"[bold]Output:[/bold] {result.get('output', 'none')[:500]}",
        title="[green]Result[/green]",
        border_style="green",
    ))

    # Check if build actually passes now
    rprint("\n[bold]Final build verification:[/bold]")
    import subprocess
    build = subprocess.run(
        "npm run build",
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=120,
        shell=True,
    )
    if build.returncode == 0:
        rprint("  [bold green]BUILD PASSES![/bold green]")
    else:
        rprint(f"  [bold red]BUILD STILL FAILS[/bold red]")
        # Show last 10 lines of stderr
        err_lines = (build.stderr or build.stdout or "").strip().split("\n")
        for line in err_lines[-10:]:
            rprint(f"    {line}")

except KeyboardInterrupt:
    rprint("\n[yellow]Test interrupted[/yellow]")
except Exception as e:
    elapsed = time.time() - start
    rprint(f"\n[red]Test failed after {elapsed:.1f}s:[/red] {e}")
    import traceback
    traceback.print_exc()
