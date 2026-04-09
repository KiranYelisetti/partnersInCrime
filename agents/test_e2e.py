"""
End-to-end test — runs a realistic feature through the full pipeline.
Tests: Orchestrator → Architect → Backend → Frontend → Reviewer → Testing
      (+ fix loop if reviewer catches mismatches)

Target: Solo Leveling HUD v2 (Next.js App Router)
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
    "[bold]Goal:[/bold] Full pipeline E2E — architect designs, backend + frontend build, "
    "reviewer checks alignment, testing verifies build.\n\n"
    "[bold]Pipeline:[/bold] architect → backend → frontend → reviewer → (fix?) → testing",
    title="[bold green]E2E Test — Full Pipeline[/bold green]",
))

# Initialize
memory, state_store = init_system()
app = build_graph()

# Realistic feature that exercises architect → backend → frontend → reviewer → testing
task = (
    "Add a Contact Us page to the Solo Leveling HUD app.\n"
    "Requirements:\n"
    "- Backend: POST /api/contact endpoint that accepts { name, email, message } "
    "and returns { success: true, id: string }\n"
    "- Frontend: /contact page with a form (name, email, message fields + submit button). "
    "On submit, call POST /api/contact and show success/error state.\n"
    "- Use the project's existing styling (Tailwind + dark theme)\n"
    "- The architect MUST write the API contract and shared TypeScript interfaces"
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
    unique_files = list(dict.fromkeys(all_files))

    rprint(Panel(
        f"[bold green]E2E Test Complete[/bold green]\n\n"
        f"[bold]Time:[/bold] {elapsed:.1f}s\n"
        f"[bold]Agents used:[/bold] {' -> '.join(agents_used)}\n"
        f"[bold]Files created/modified ({len(unique_files)} unique):[/bold]\n"
        + ("\n".join(f"  - {f}" for f in unique_files) if unique_files else "  none") + "\n"
        f"[bold]Errors:[/bold] {result.get('error', 'none')}\n"
        f"[bold]Output:[/bold] {result.get('output', 'none')[:500]}",
        title="[green]Result[/green]",
        border_style="green",
    ))

    # Verify key files
    rprint("\n[bold]File verification:[/bold]")
    expected = [
        PROJECT_ROOT / "docs" / "architecture" / "api-contract.json",
        PROJECT_ROOT / "src" / "lib" / "types" / "api.ts",
        PROJECT_ROOT / "src" / "app" / "api" / "contact" / "route.ts",
        PROJECT_ROOT / "src" / "app" / "contact" / "page.tsx",
    ]
    for f in expected:
        if f.exists():
            rprint(f"  [green]EXISTS[/green] {f} ({f.stat().st_size} bytes)")
        else:
            rprint(f"  [yellow]MISSING[/yellow] {f}")

    # Also check any other files the agents created
    for f in unique_files:
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
