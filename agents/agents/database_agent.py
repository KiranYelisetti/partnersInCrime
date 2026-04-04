"""
Database Agent — Data models, schemas, ORM setup, DB connection logic.
Loads config from definitions/agents.yaml.
"""
from state import AgentState
from agents.agent_base import run_agent_loop
from definitions.loader import get_agent_config, resolve_tools

_config = get_agent_config("database")
_tools = resolve_tools(_config["tools"])

_memory = None
_state_store = None


def set_dependencies(memory=None, state_store=None):
    global _memory, _state_store
    _memory = memory
    _state_store = state_store


def database_node(state: AgentState) -> dict:
    return run_agent_loop(
        state=state,
        agent_name="database",
        system_prompt=_config["system_prompt"],
        tools=_tools,
        memory=_memory,
        state_store=_state_store,
    )
