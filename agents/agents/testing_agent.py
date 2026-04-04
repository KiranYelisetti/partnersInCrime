"""
Testing Agent — QA engineer that builds, tests, and reports bugs.

Two modes (selected by fix_round in state):
1. REPORT mode (fix_round == 0): Run build, collect errors, write structured bug report.
2. VERIFY mode (fix_round > 0): Re-run build to check if fixes worked.

Loads config from definitions/agents.yaml.
"""
from state import AgentState
from agents.agent_base import run_agent_loop
from definitions.loader import get_agent_config, resolve_tools

_config = get_agent_config("testing")
_tools = resolve_tools(_config["tools"])

_memory = None
_state_store = None


def set_dependencies(memory=None, state_store=None):
    global _memory, _state_store
    _memory = memory
    _state_store = state_store


def testing_node(state: AgentState) -> dict:
    # Choose prompt based on fix round
    fix_round = state.get("_fix_round", 0)
    if fix_round > 0:
        prompt = _config["system_prompt_verify"]
    else:
        prompt = _config["system_prompt_report"]

    return run_agent_loop(
        state=state,
        agent_name="testing",
        system_prompt=prompt,
        tools=_tools,
        memory=_memory,
        state_store=_state_store,
    )
