"""Pipeline stage machine.

Enforces the spec's stage ordering in code: a stage can only be recorded if it
is the next stage in the sequence, and downstream steps can assert that their
prerequisites completed. State is persisted to pipeline_state.json after every
transition so validation can verify ordering from disk.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

STAGES = [
    "INIT",
    "INPUTS_LOADED",
    "DOCUMENTS_CHUNKED",
    "INDEX_BUILT",
    "RETRIEVAL_COMPLETE",
    "DRAFT_ANSWERS_GENERATED",
    "HUMAN_REVIEW_COMPLETE",
    "ANSWERS_AUDITED",
    "FINAL_REPORT_GENERATED",
    "VALIDATION_COMPLETE",
    "RESULTS_FINALISED",
]

STATE_FILE = "pipeline_state.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class StageOrderError(RuntimeError):
    pass


class StageMachine:
    def __init__(self, root: Path, mode: str, provider: str):
        self.root = Path(root)
        self.state: dict = {
            "mode": mode,
            "provider": provider,
            "inputs": {},
            "stages": [],
        }

    @property
    def path(self) -> Path:
        return self.root / STATE_FILE

    def record(self, stage: str) -> None:
        """Record a stage transition. Raises if `stage` is not the next stage."""
        idx = len(self.state["stages"])
        if idx >= len(STAGES) or STAGES[idx] != stage:
            expected = STAGES[idx] if idx < len(STAGES) else "<none>"
            raise StageOrderError(
                f"Stage order violation: expected {expected!r}, got {stage!r}"
            )
        self.state["stages"].append({"stage": stage, "timestamp": utc_now()})
        self._flush()

    def completed(self, stage: str) -> bool:
        return any(s["stage"] == stage for s in self.state["stages"])

    def require(self, *stages: str) -> None:
        missing = [s for s in stages if not self.completed(s)]
        if missing:
            raise StageOrderError(f"Required stage(s) not complete: {missing}")

    def record_inputs(
        self, documents_dir: Path, queries_path: Path, policy_path: Path
    ) -> None:
        docs = {
            p.name: sha256_file(p) for p in sorted(documents_dir.glob("*.txt"))
        }
        self.state["inputs"] = {
            "documents_dir": str(documents_dir),
            "documents": docs,
            "queries": {
                "path": str(queries_path),
                "sha256": sha256_file(queries_path),
            },
            "policy": {
                "path": str(policy_path),
                "sha256": sha256_file(policy_path),
            },
        }
        self._flush()

    def set_provider(self, provider: str, model: str) -> None:
        self.state["provider"] = provider
        self.state["model"] = model
        self._flush()

    def _flush(self) -> None:
        self.path.write_text(
            json.dumps(self.state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
