"""
Run the Solo Leveling HUD v2 build — Step 1: Architect designs the full spec.

This kicks off the agent system targeting the v2 project directory.
Agents will read the v1 codebase via reference tools and build v2 from scratch.
"""
import sys
import os
import time
from pathlib import Path

# Fix Windows Unicode output
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))

from rich import print as rprint
from rich.panel import Panel

from config import PROJECT_ROOT, REFERENCE_PROJECT_ROOT
from main import init_system, build_graph

rprint(Panel(
    f"[bold]Project root (v2):[/bold] {PROJECT_ROOT}\n"
    f"[bold]Reference (v1):[/bold] {REFERENCE_PROJECT_ROOT}\n\n"
    "Step 1: Architect agent designs the full technical spec.\n"
    "It will explore v1 code, then write a design doc to docs/architecture/.",
    title="[bold green]Solo Leveling HUD v2 — Agent Build[/bold green]",
))

# Initialize
memory, state_store = init_system()

# Build graph
app = build_graph()

# The task — multi-agent build
task = (
    "Build Solo Leveling HUD v2 with Next.js App Router, TypeScript, MongoDB (Mongoose), "
    "Firebase Auth, and Razorpay. This is a gamified productivity app with 7 life domains "
    "(career, business, finance, physical, mental, community, fuel). "
    "Read the v1 reference project for context. Steps:\n"
    "1. Infra: scaffold with create-next-app, install mongoose/firebase-admin/razorpay deps, "
    "create .env.example\n"
    "2. Architect: read v1 code + scaffolded project, design full technical spec, write to docs/architecture/\n"
    "3. Database: create Mongoose models for User, all 7 domain item types, Subscription\n"
    "4. Backend: create Next.js API routes for auth, generic CRUD per domain, activity logs, payments\n"
    "5. Frontend: create React components for dashboard, domain pages, auth pages, sidebar\n"
    "6. Testing: write and run vitest tests, verify npm run build passes"
)

rprint(f"\n[bold cyan]Task:[/bold cyan] {task[:200]}...\n{'=' * 60}")

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
    })

    elapsed = time.time() - start
    agents_used = [m["role"] for m in result.get("messages", [])
                  if m["role"] not in ("orchestrator", "human")]
    all_files = result.get("files_changed", [])

    rprint(Panel(
        f"[bold green]Build step complete[/bold green]\n\n"
        f"[bold]Time:[/bold] {elapsed:.1f}s\n"
        f"[bold]Agents used:[/bold] {' -> '.join(agents_used)}\n"
        f"[bold]Files created:[/bold]\n"
        + ("\n".join(f"  - {f}" for f in all_files) if all_files else "  none") + "\n"
        f"[bold]Errors:[/bold] {result.get('error', 'none')}\n"
        f"[bold]Output:[/bold] {result.get('output', 'none')[:500]}",
        title="[green]Result[/green]",
        border_style="green",
    ))

    # Verify files exist
    if all_files:
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
    rprint("\n[yellow]Build interrupted[/yellow]")
except Exception as e:
    rprint(f"\n[red]Build failed:[/red] {e}")
    import traceback
    traceback.print_exc()
