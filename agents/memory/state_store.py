"""
State store — shared state and task queue.
Uses Redis if available, falls back to in-memory dict.

This enables:
- Task queuing (for Phase 5: 24/7 operation)
- Shared context between agents
- Audit trail of all agent actions
"""
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from collections import deque

from config import REDIS_URL


class StateStore:
    """
    Shared state manager with Redis backend (or in-memory fallback).
    """

    def __init__(self, redis_url: str = None):
        self._redis = None
        self._memory: Dict[str, Any] = {}
        self._task_queue: deque = deque()
        self._action_log: List[Dict] = []

        # Try connecting to Redis
        url = redis_url or REDIS_URL
        try:
            import redis as redis_lib
            self._redis = redis_lib.from_url(url, decode_responses=True)
            self._redis.ping()
            self._mode = "redis"
        except Exception:
            self._redis = None
            self._mode = "memory"

    @property
    def mode(self) -> str:
        return self._mode

    # ── Task Queue ────────────────────────────────────────────────

    def push_task(self, task: Dict[str, Any]) -> int:
        """
        Add a task to the queue. Returns queue length.
        Task dict should have at least: {"description": "...", "priority": 1}
        """
        task["queued_at"] = datetime.now().isoformat()

        if self._redis:
            self._redis.rpush("task_queue", json.dumps(task))
            return self._redis.llen("task_queue")
        else:
            self._task_queue.append(task)
            return len(self._task_queue)

    def pop_task(self) -> Optional[Dict[str, Any]]:
        """Get the next task from the queue (FIFO)."""
        if self._redis:
            raw = self._redis.lpop("task_queue")
            return json.loads(raw) if raw else None
        else:
            return self._task_queue.popleft() if self._task_queue else None

    def peek_tasks(self, count: int = 5) -> List[Dict[str, Any]]:
        """View upcoming tasks without removing them."""
        if self._redis:
            raw_list = self._redis.lrange("task_queue", 0, count - 1)
            return [json.loads(r) for r in raw_list]
        else:
            return list(self._task_queue)[:count]

    def queue_length(self) -> int:
        if self._redis:
            return self._redis.llen("task_queue")
        return len(self._task_queue)

    # ── Shared Context ────────────────────────────────────────────

    def set_context(self, key: str, value: Any) -> None:
        """Store a shared context value that any agent can read."""
        if self._redis:
            self._redis.hset("shared_context", key, json.dumps(value))
        else:
            self._memory[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Retrieve a shared context value."""
        if self._redis:
            raw = self._redis.hget("shared_context", key)
            return json.loads(raw) if raw else default
        else:
            return self._memory.get(key, default)

    def get_all_context(self) -> Dict[str, Any]:
        """Get all shared context."""
        if self._redis:
            raw = self._redis.hgetall("shared_context")
            return {k: json.loads(v) for k, v in raw.items()}
        else:
            return dict(self._memory)

    def delete_context(self, key: str) -> None:
        if self._redis:
            self._redis.hdel("shared_context", key)
        else:
            self._memory.pop(key, None)

    # ── Audit Trail ───────────────────────────────────────────────

    def log_agent_action(
        self,
        agent: str,
        action: str,
        result: str,
        success: bool = True,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Log an agent action for audit/debugging.
        """
        entry = {
            "agent": agent,
            "action": action,
            "result": result[:1000],
            "success": success,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {}),
        }

        if self._redis:
            self._redis.rpush("action_log", json.dumps(entry))
            # Keep only last 1000 entries
            self._redis.ltrim("action_log", -1000, -1)
        else:
            self._action_log.append(entry)
            self._action_log = self._action_log[-1000:]

    def get_action_log(self, count: int = 20, agent: Optional[str] = None) -> List[Dict]:
        """Get recent action log entries, optionally filtered by agent."""
        if self._redis:
            raw_list = self._redis.lrange("action_log", -count, -1)
            entries = [json.loads(r) for r in raw_list]
        else:
            entries = self._action_log[-count:]

        if agent:
            entries = [e for e in entries if e.get("agent") == agent]

        return entries

    # ── Status ────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get overall state store status."""
        log_count = (
            self._redis.llen("action_log") if self._redis
            else len(self._action_log)
        )
        return {
            "mode": self._mode,
            "queue_length": self.queue_length(),
            "action_log_entries": log_count,
            "context_keys": (
                list(self._redis.hkeys("shared_context")) if self._redis
                else list(self._memory.keys())
            ),
        }
