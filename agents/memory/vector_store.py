"""
Vector store / RAG memory layer using ChromaDB + Ollama embeddings.

This is the "learn from mistakes" engine:
- Embeds your codebase so agents have project context
- Stores task outcomes (successes and failures)
- Retrieves relevant past context before each new task
"""
import os
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import chromadb
from chromadb.config import Settings

from config import CHROMADB_PATH, OLLAMA_BASE_URL, EMBEDDING_MODEL


class OllamaEmbeddingFunction:
    """
    Custom embedding function that calls Ollama's API.
    ChromaDB calls this to turn text → vectors.
    """
    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or EMBEDDING_MODEL
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")

    def name(self) -> str:
        return f"ollama-{self.model}"

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._embed(input)

    def embed_documents(self, documents: List[str] = None, **kwargs) -> List[List[float]]:
        # ChromaDB v1.5 may pass input= as keyword
        texts = documents or kwargs.get("input", [])
        return self._embed(texts)

    def embed_query(self, query: str = None, **kwargs) -> List[float]:
        # ChromaDB v1.5 may pass input= as keyword
        text = query or kwargs.get("input", "")
        if isinstance(text, list):
            return self._embed(text)
        return self._embed([text])[0]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        import requests
        embeddings = []
        for text in texts:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])
        return embeddings


class AgentMemory:
    """
    Manages three ChromaDB collections:
    1. codebase — chunks of your project files
    2. task_results — outcomes of past tasks (success/failure)
    3. mistakes — failure patterns with fixes
    """

    # File extensions we'll embed from the codebase
    CODE_EXTENSIONS = {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css",
        ".json", ".yaml", ".yml", ".toml", ".md", ".sql",
        ".dockerfile", ".sh", ".bat", ".ps1",
    }

    # Directories to skip during codebase embedding
    SKIP_DIRS = {
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        "chromadb_data", ".next", "dist", "build", ".mypy_cache",
    }

    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or CHROMADB_PATH
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.embed_fn = OllamaEmbeddingFunction()

        # Initialize collections
        self.codebase = self.client.get_or_create_collection(
            name="codebase",
            embedding_function=self.embed_fn,
            metadata={"description": "Project source code chunks"},
        )
        self.task_results = self.client.get_or_create_collection(
            name="task_results",
            embedding_function=self.embed_fn,
            metadata={"description": "Past task outcomes for learning"},
        )
        self.mistakes = self.client.get_or_create_collection(
            name="mistakes",
            embedding_function=self.embed_fn,
            metadata={"description": "Failure patterns and fixes"},
        )

    # ── Codebase Embedding ────────────────────────────────────────

    def _chunk_file(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split file content into overlapping chunks."""
        lines = content.splitlines()
        chunks = []
        i = 0
        while i < len(lines):
            chunk_lines = lines[i:i + chunk_size]
            chunk_text = "\n".join(chunk_lines)
            if chunk_text.strip():
                chunks.append(chunk_text)
            i += chunk_size - overlap
        return chunks if chunks else [content[:5000]]  # fallback for small files

    def _file_hash(self, content: str) -> str:
        """Hash file content to detect changes."""
        return hashlib.md5(content.encode()).hexdigest()

    def embed_codebase(self, root_dir: str, force: bool = False) -> Dict[str, int]:
        """
        Walk the project directory, chunk files, and embed them.
        Only re-embeds files that changed since last run (unless force=True).

        Returns: {"files_processed": N, "chunks_added": N, "skipped": N}
        """
        root = Path(root_dir)
        stats = {"files_processed": 0, "chunks_added": 0, "skipped": 0}

        for filepath in root.rglob("*"):
            # Skip unwanted directories
            if any(skip in filepath.parts for skip in self.SKIP_DIRS):
                continue
            if not filepath.is_file():
                continue
            if filepath.suffix.lower() not in self.CODE_EXTENSIONS:
                continue
            if filepath.stat().st_size > 500_000:  # Skip files > 500KB
                continue

            try:
                content = filepath.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue

            file_hash = self._file_hash(content)
            rel_path = str(filepath.relative_to(root))

            # Check if already embedded with same hash
            if not force:
                existing = self.codebase.get(
                    where={"file_path": rel_path},
                    include=["metadatas"],
                )
                if existing["ids"] and existing["metadatas"]:
                    if existing["metadatas"][0].get("file_hash") == file_hash:
                        stats["skipped"] += 1
                        continue
                    # File changed — delete old chunks first
                    self.codebase.delete(ids=existing["ids"])

            # Chunk and embed
            chunks = self._chunk_file(content, chunk_size=50, overlap=10)  # ~50 lines per chunk
            for i, chunk in enumerate(chunks):
                doc_id = f"{rel_path}::chunk_{i}"
                self.codebase.upsert(
                    ids=[doc_id],
                    documents=[f"File: {rel_path}\n\n{chunk}"],
                    metadatas=[{
                        "file_path": rel_path,
                        "chunk_index": i,
                        "file_hash": file_hash,
                        "embedded_at": datetime.now().isoformat(),
                    }],
                )
                stats["chunks_added"] += 1

            stats["files_processed"] += 1

        return stats

    # ── Task Result Logging ───────────────────────────────────────

    def embed_task_result(
        self,
        task: str,
        agent: str,
        result: str,
        success: bool,
        artifacts: Optional[List[str]] = None,
    ) -> str:
        """
        Store the outcome of a task so agents can learn from past work.
        Returns the document ID.
        """
        doc_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{agent}"
        status = "SUCCESS" if success else "FAILURE"

        document = (
            f"Task: {task}\n"
            f"Agent: {agent}\n"
            f"Status: {status}\n"
            f"Result: {result[:2000]}\n"
            f"Artifacts: {', '.join(artifacts or [])}"
        )

        self.task_results.upsert(
            ids=[doc_id],
            documents=[document],
            metadatas=[{
                "task": task[:500],
                "agent": agent,
                "success": success,
                "timestamp": datetime.now().isoformat(),
            }],
        )
        return doc_id

    # ── Mistake Logging ───────────────────────────────────────────

    def embed_mistake(
        self,
        task: str,
        agent: str,
        error: str,
        fix: str = "",
        severity: int = 1,
        error_type: str = "",
        file_path: str = "",
    ) -> str:
        """
        Store a failure pattern so agents avoid repeating it.
        severity: 1=minor, 2=moderate, 3=critical
        error_type: classified error (e.g. "ImportError", "TypeError", "build_failure")
        file_path: the file where the error occurred
        """
        doc_id = f"mistake_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{agent}"

        # Build a structured, searchable document
        parts = [f"MISTAKE by {agent}"]
        if error_type:
            parts.append(f"Error type: {error_type}")
        parts.append(f"Task: {task}")
        parts.append(f"Error: {error}")
        if file_path:
            parts.append(f"File: {file_path}")
        if fix:
            parts.append(f"Fix applied: {fix}")
        parts.append(f"Severity: {severity}")
        document = "\n".join(parts)

        self.mistakes.upsert(
            ids=[doc_id],
            documents=[document],
            metadatas=[{
                "task": task[:500],
                "agent": agent,
                "error": error[:500],
                "fix": fix[:500],
                "severity": severity,
                "error_type": error_type[:100],
                "file_path": file_path[:300],
                "timestamp": datetime.now().isoformat(),
            }],
        )
        return doc_id

    # ── Retrieval ─────────────────────────────────────────────────

    def retrieve_context(self, query: str, k: int = 5) -> str:
        """
        Get relevant codebase chunks for a task.
        Returns formatted context string for injection into agent prompts.
        """
        if self.codebase.count() == 0:
            return "(No codebase context available — run embed_codebase first)"

        results = self.codebase.query(
            query_texts=[query],
            n_results=min(k, self.codebase.count()),
        )

        if not results["documents"][0]:
            return "(No relevant codebase context found)"

        chunks = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append(f"--- {meta.get('file_path', 'unknown')} ---\n{doc}")

        return "CODEBASE CONTEXT:\n\n" + "\n\n".join(chunks)

    def retrieve_past_results(self, query: str, k: int = 3) -> str:
        """Get past task results similar to the current task."""
        if self.task_results.count() == 0:
            return ""

        results = self.task_results.query(
            query_texts=[query],
            n_results=min(k, self.task_results.count()),
        )

        if not results["documents"][0]:
            return ""

        return "PAST TASK RESULTS:\n\n" + "\n\n".join(results["documents"][0])

    def retrieve_mistakes(self, query: str, k: int = 3) -> str:
        """
        Get past mistakes similar to the current task.
        This is the "learn from mistakes" retrieval — injected into agent prompts
        so they avoid repeating known failures.
        """
        if self.mistakes.count() == 0:
            return ""

        results = self.mistakes.query(
            query_texts=[query],
            n_results=min(k, self.mistakes.count()),
        )

        if not results["documents"][0]:
            return ""

        # Format mistakes as actionable warnings with fix instructions
        formatted = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            fix = meta.get("fix", "")
            error_type = meta.get("error_type", "")
            header = f"[{error_type}]" if error_type else "[error]"
            if fix:
                formatted.append(f"{header} {doc}\n--> WHAT WORKED: {fix}")
            else:
                formatted.append(f"{header} {doc}")

        return (
            "PAST MISTAKES (avoid repeating these):\n\n"
            + "\n---\n".join(formatted)
        )

    # ── Utility ───────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        """Return counts for each collection."""
        return {
            "codebase_chunks": self.codebase.count(),
            "task_results": self.task_results.count(),
            "mistakes": self.mistakes.count(),
        }
