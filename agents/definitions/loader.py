"""
Agent definition loader — reads agents.yaml as the single source of truth
for system prompts, timeouts, tool sets, and write scopes.
"""
import yaml
from pathlib import Path
from typing import Dict, List

_YAML_PATH = Path(__file__).parent / "agents.yaml"
_cache = None


def _load_yaml() -> dict:
    """Load and cache the YAML config."""
    global _cache
    if _cache is None:
        with open(_YAML_PATH, "r", encoding="utf-8") as f:
            _cache = yaml.safe_load(f)
    return _cache


def get_agent_config(agent_name: str) -> dict:
    """Get the full configuration dict for a specific agent."""
    configs = _load_yaml()
    if agent_name not in configs:
        raise KeyError(f"Agent '{agent_name}' not found in agents.yaml")
    return configs[agent_name]


def get_all_write_scopes() -> Dict[str, List[str]]:
    """
    Get write scopes for all agents.
    Used by hooks.py as the single source of truth for path restrictions.
    """
    configs = _load_yaml()
    return {
        name: cfg.get("write_scopes", [])
        for name, cfg in configs.items()
    }


def resolve_tools(tool_spec: str) -> list:
    """
    Resolve a YAML tool specification string (e.g. "ALL_TOOLS",
    "FILE_TOOLS + WEB_TOOLS + REFERENCE_TOOLS") to actual tool objects.

    Lazy-imports from the tools package to avoid circular imports
    (hooks.py -> loader.py -> tools -> file_tools -> hooks.py).
    """
    from tools import ALL_TOOLS
    from tools.file_tools import FILE_TOOLS, REFERENCE_TOOLS
    from tools.web_tools import WEB_TOOLS

    registry = {
        "ALL_TOOLS": ALL_TOOLS,
        "FILE_TOOLS": FILE_TOOLS,
        "WEB_TOOLS": WEB_TOOLS,
        "REFERENCE_TOOLS": REFERENCE_TOOLS,
    }

    # Single token — direct lookup
    spec_stripped = tool_spec.strip()
    if spec_stripped in registry:
        return registry[spec_stripped]

    # Composite — "FILE_TOOLS + WEB_TOOLS + REFERENCE_TOOLS"
    parts = [p.strip() for p in spec_stripped.split("+")]
    tools = []
    for part in parts:
        if part not in registry:
            raise ValueError(f"Unknown tool set in agents.yaml: '{part}'")
        tools.extend(registry[part])
    return tools
