"""LLM provider abstraction with per-call logging to llm_calls.jsonl.

Provider auto-detection order (override with RAG_LLM_PROVIDER=anthropic|claude_cli|openai):
  1. anthropic   — ANTHROPIC_API_KEY set and `anthropic` SDK importable (model claude-opus-5)
  2. claude_cli  — `claude` CLI on PATH (headless `claude -p`, subscription auth)
  3. openai      — OPENAI_API_KEY set (stdlib urllib, no SDK required)

Every call appends one JSONL record: stage, query_id, timestamp, provider,
model, prompt_hash, input_artifacts, output_artifact.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

from .state import sha256_text, utc_now

LLM_CALLS_FILE = "llm_calls.jsonl"
ANTHROPIC_MODEL = os.environ.get("RAG_ANTHROPIC_MODEL", "claude-opus-5")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
CALL_TIMEOUT_S = 300


class LLMError(RuntimeError):
    pass


def detect_provider() -> tuple[str, str]:
    """Return (provider, model). Honors RAG_LLM_PROVIDER override."""
    forced = os.environ.get("RAG_LLM_PROVIDER")
    candidates = [forced] if forced else ["anthropic", "claude_cli", "openai"]
    for name in candidates:
        if name == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic  # noqa: F401
                return "anthropic", ANTHROPIC_MODEL
            except ImportError:
                if forced:
                    raise LLMError(
                        "RAG_LLM_PROVIDER=anthropic but the `anthropic` SDK is "
                        "not installed (pip install anthropic)"
                    )
        elif name == "claude_cli" and shutil.which("claude"):
            model = os.environ.get("RAG_CLAUDE_CLI_MODEL", "default")
            return "claude_cli", model
        elif name == "openai" and os.environ.get("OPENAI_API_KEY"):
            return "openai", OPENAI_MODEL
    raise LLMError(
        "No LLM provider available. Set ANTHROPIC_API_KEY (+ pip install "
        "anthropic), install/authenticate the `claude` CLI, or set "
        "OPENAI_API_KEY. Force one with RAG_LLM_PROVIDER."
    )


# --------------------------------------------------------------------------
# Provider backends — each takes a prompt, returns raw response text.
# --------------------------------------------------------------------------

def _call_anthropic(prompt: str, schema: dict) -> str:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=16000,
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": prompt}],
    )
    return next(b.text for b in response.content if b.type == "text")


def _call_claude_cli(prompt: str, _schema: dict) -> str:
    cmd = ["claude", "-p", prompt, "--output-format", "json"]
    model = os.environ.get("RAG_CLAUDE_CLI_MODEL")
    if model:
        cmd += ["--model", model]
    # stdin=DEVNULL is load-bearing: the pipeline's own stdin carries the
    # human-review override lines, which the CLI child must not consume.
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=CALL_TIMEOUT_S,
        stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        raise LLMError(f"claude CLI failed (rc={proc.returncode}): {proc.stderr[:500]}")
    envelope = json.loads(proc.stdout)
    if envelope.get("is_error"):
        raise LLMError(f"claude CLI returned error result: {envelope.get('result', '')[:500]}")
    return envelope.get("result", "")


def _call_openai(prompt: str, _schema: dict) -> str:
    body = json.dumps(
        {
            "model": OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        },
    )
    with urllib.request.urlopen(req, timeout=CALL_TIMEOUT_S) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


_BACKENDS = {
    "anthropic": _call_anthropic,
    "claude_cli": _call_claude_cli,
    "openai": _call_openai,
}


# --------------------------------------------------------------------------
# JSON extraction + repair retry
# --------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        # strip a ```json ... ``` fence
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


class LLMClient:
    def __init__(self, root: Path, provider: str, model: str):
        self.root = Path(root)
        self.provider = provider
        self.model = model

    def _log(self, stage: str, query_id: str | None, prompt: str,
             input_artifacts: list[str], output_artifact: str) -> None:
        record = {
            "stage": stage,
            "query_id": query_id,
            "timestamp": utc_now(),
            "provider": self.provider,
            "model": self.model,
            "prompt_hash": sha256_text(prompt),
            "input_artifacts": input_artifacts,
            "output_artifact": output_artifact,
        }
        with (self.root / LLM_CALLS_FILE).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def call_json(
        self,
        *,
        stage: str,
        query_id: str | None,
        prompt: str,
        required_keys: list[str],
        schema: dict,
        input_artifacts: list[str],
        output_artifact: str,
    ) -> dict:
        """Make one LLM call, log it, parse a JSON object with required_keys.
        One repair retry (also logged) on parse/shape failure."""
        backend = _BACKENDS[self.provider]

        attempt_prompt = prompt
        last_error = None
        for attempt in (1, 2):
            raw = backend(attempt_prompt, schema)
            self._log(stage, query_id, attempt_prompt, input_artifacts, output_artifact)
            try:
                obj = _extract_json(raw)
                missing = [k for k in required_keys if k not in obj]
                if missing:
                    raise LLMError(f"missing keys {missing}")
                return obj
            except (json.JSONDecodeError, LLMError) as exc:
                last_error = exc
                attempt_prompt = (
                    prompt
                    + "\n\nYour previous output was invalid ("
                    + str(exc)[:200]
                    + "). Return ONLY a single valid JSON object with exactly "
                    + "these keys: "
                    + ", ".join(required_keys)
                    + ". No prose, no markdown fences."
                )
        raise LLMError(
            f"LLM output invalid after retry for stage={stage} "
            f"query_id={query_id}: {last_error}"
        )
