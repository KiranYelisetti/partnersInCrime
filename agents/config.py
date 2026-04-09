"""
Centralized configuration for the Partners in Crime agent system.
All settings loaded from .env — agents import from here instead of hardcoding.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (same directory as this file)
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(_ENV_PATH)


# ── LLM Provider ────────────────────────────────────────────────
# "claude" = Anthropic API (requires ANTHROPIC_API_KEY)
# "groq"   = Groq Cloud API (requires GROQ_API_KEY) — free tier available
# "ollama" = local Ollama (default)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# ── Ollama / Model settings ─────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
SPECIALIST_MODEL = os.getenv("SPECIALIST_MODEL", "qwen2.5-coder:14b")
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "qwen2.5-coder:14b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# Context window size — 16384 needed for agentic tool-calling (multi-turn tool history).
# 14B model + 16K context fits in ~11GB VRAM on RTX 5070.
NUM_CTX = int(os.getenv("NUM_CTX", "16384"))

# Keep-alive time for models in Ollama (seconds). 0 = unload immediately after use.
# Setting to 0 frees VRAM between calls, preventing the embedding model and LLM
# from fighting for GPU memory. Trade-off: ~2-3s cold start per call.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "120s")

# Cooldown seconds between LLM calls to let Ollama reclaim VRAM
LLM_COOLDOWN = float(os.getenv("LLM_COOLDOWN", "2.0"))

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.getenv(
    "PROJECT_ROOT",
    str(Path(__file__).parent.parent)
))

CHROMADB_PATH = os.getenv(
    "CHROMADB_PATH",
    str(Path(__file__).parent / "chromadb_data")
)

OUTPUT_DIR = Path(os.getenv(
    "OUTPUT_DIR",
    str(Path(__file__).parent / "output")
))

# ── Redis ────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# ── Agent behavior ──────────────────────────────────────────────
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
COMMAND_TIMEOUT = int(os.getenv("COMMAND_TIMEOUT", "30"))

# Allowed shell commands
_allowed = os.getenv("ALLOWED_COMMANDS", "python,pip,pytest,npm,npx,node,tsc,git,ls,dir,cat,type,echo,mkdir,cd")
ALLOWED_COMMANDS = set(cmd.strip() for cmd in _allowed.split(","))

MAX_FILE_READ_BYTES = int(os.getenv("MAX_FILE_READ_BYTES", str(1024 * 1024)))
MAX_COMMAND_OUTPUT = int(os.getenv("MAX_COMMAND_OUTPUT", "2000"))

# ── Agentic execution ─────────────────────────────────────────
# Max tool-calling iterations per agent before forcing stop
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "25"))

# Max chars to keep from each tool result (saves context window)
MAX_TOOL_RESULT_CHARS = int(os.getenv("MAX_TOOL_RESULT_CHARS", "3000"))

# Build verification: auto-run build check after agent calls task_done.
# If it fails, agent re-enters the loop to fix the issue.
BUILD_VERIFY_ENABLED = os.getenv("BUILD_VERIFY_ENABLED", "true").lower() == "true"
BUILD_VERIFY_MAX_RETRIES = int(os.getenv("BUILD_VERIFY_MAX_RETRIES", "2"))

# Wall-clock timeout per agent (seconds). Kills agent if it exceeds this,
# regardless of iteration count. Prevents runaway agents.
AGENT_TIMEOUT_SECONDS = int(os.getenv("AGENT_TIMEOUT_SECONDS", "600"))  # 10 min default

# Cycle detection: if an agent calls the same tool with the same args
# this many times in a row, force-stop it. Catches infinite loops.
MAX_IDENTICAL_TOOL_CALLS = int(os.getenv("MAX_IDENTICAL_TOOL_CALLS", "3"))

# Integration fix loop: after all agents build, tester reports bugs.
# Orchestrator routes fixes back to responsible agents, then retests.
# This repeats up to MAX_FIX_ROUNDS times.
MAX_FIX_ROUNDS = int(os.getenv("MAX_FIX_ROUNDS", "3"))

# Logging: per-agent structured logs for audit trail
AGENT_LOG_DIR = Path(os.getenv("AGENT_LOG_DIR", str(Path(__file__).parent / "logs")))

# Reference project: agents can read files from this path (read-only)
# to understand the v1 codebase when building v2.
_ref_root = os.getenv("REFERENCE_PROJECT_ROOT", "")
REFERENCE_PROJECT_ROOT = Path(_ref_root) if _ref_root else None

# RAG retrieval limits — fewer chunks = fewer embedding calls = less VRAM pressure
RAG_CODEBASE_K = int(os.getenv("RAG_CODEBASE_K", "0"))
RAG_MISTAKES_K = int(os.getenv("RAG_MISTAKES_K", "2"))
RAG_RESULTS_K = int(os.getenv("RAG_RESULTS_K", "2"))


def get_llm(role: str = "specialist"):
    """
    Factory function returning an LLM instance configured for the role.
    Supports both Claude (Anthropic API) and Ollama (local) via LLM_PROVIDER env var.

    Orchestrator gets lower max_tokens since it only plans (no tool history).
    Specialists get more tokens for multi-turn tool calling.
    """
    model = ORCHESTRATOR_MODEL if role == "orchestrator" else SPECIALIST_MODEL
    temperature = 0.2 if role == "orchestrator" else 0.1

    if LLM_PROVIDER == "claude":
        from langchain_anthropic import ChatAnthropic

        max_tokens = 4096 if role == "orchestrator" else 8192

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=120,
        )

    elif LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        groq_api_key = os.getenv("GROQ_API_KEY", "")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not set in .env")

        # Free tier has TPM limits — keep output concise
        max_tokens = 2048 if role == "orchestrator" else 4096

        return ChatGroq(
            model=model,
            api_key=groq_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    else:
        from langchain_ollama import ChatOllama

        ctx = NUM_CTX

        # Disable reasoning/thinking mode for qwen3 models.
        # Without this, qwen3 generates thousands of hidden CoT tokens
        # before outputting JSON, causing 10+ minute hangs.
        is_thinking_model = "qwen3" in model.lower()

        return ChatOllama(
            model=model,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
            num_ctx=ctx,
            keep_alive=OLLAMA_KEEP_ALIVE,
            reasoning=False if is_thinking_model else None,
        )
