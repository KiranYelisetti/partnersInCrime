"""
Shared state for the LangGraph agent system.
All nodes read and write to this TypedDict.
"""
from typing import TypedDict, Annotated, List, Optional
from operator import add


class AgentState(TypedDict):
    # The task description coming in
    task: str

    # Which agent to route to next
    next_agent: Optional[str]

    # Accumulated messages/results from all agents
    messages: Annotated[List[dict], add]

    # Final output artifact (summary text from last agent)
    output: Optional[str]

    # Error state for retry logic
    error: Optional[str]
    retry_count: int

    # Whether orchestrator needs human input before continuing
    needs_human: bool

    # RAG-retrieved context injected by orchestrator
    context: Optional[str]

    # Multi-step agent plan from orchestrator (e.g., ["database", "backend", "testing"])
    agent_plan: Optional[List[str]]

    # Current step index in the agent plan
    current_step: int

    # Per-agent subtask breakdown: {"database": "Create Task model...", "backend": "Create CRUD..."}
    _plan_details: Optional[dict]

    # Files created/modified by agents — passed to next agent for context
    files_changed: Annotated[List[str], add]
