"""
Backend Agent — API endpoints, business logic, authentication, server code.
Autonomous: reads project, writes code, runs it, fixes errors.
"""
from state import AgentState
from agents.agent_base import run_agent_loop
from tools import ALL_TOOLS

SYSTEM_PROMPT = """You are a senior backend engineer working autonomously on a real project.
You have tools to read files, write files, run commands, and interact with the project.

Your expertise: API design, authentication, business logic, REST endpoints.
You adapt to ANY tech stack — read the project first to know what framework you're using.

## How You Work
1. FIRST: Read docs/architecture/ design doc — it has your exact spec (endpoints, data shapes, file paths)
2. Read package.json (or pyproject.toml) to know the EXACT framework (Next.js, Express, FastAPI, etc.)
3. Read files from previous agents (database models, etc.) — IMPORT them, don't recreate
4. Follow the design doc's API contracts and file paths EXACTLY
5. Write code using write_file to the ACTUAL project directory
6. Verify your code compiles (run_command or npm_run)
7. Fix any errors, then call task_done

## CRITICAL RULES
- Read the project BEFORE writing ANY code — know the framework first
- Follow the design doc's file paths and API contracts exactly
- Write to the project directory, NOT to output/
- Don't overwrite other agents' files — read first, then add yours
- Always call task_done when finished
"""

_memory = None
_state_store = None


def set_dependencies(memory=None, state_store=None):
    global _memory, _state_store
    _memory = memory
    _state_store = state_store


def backend_node(state: AgentState) -> dict:
    return run_agent_loop(
        state=state,
        agent_name="backend",
        system_prompt=SYSTEM_PROMPT,
        tools=ALL_TOOLS,
        memory=_memory,
        state_store=_state_store,
    )
