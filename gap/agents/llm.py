"""One LLM interface for every agent. Caching/logging/budget OWNER: done

Every agent calls complete() and nothing else, so swapping the model or
adapter is one config change. Every call is logged to the run folder: those
logs become SFT data in Phase 6.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from gap.utils.runs import RunDir

REPO = Path(__file__).resolve().parents[2]


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    latency_s: float = 0.0


class Backend(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict], *, temperature: float,
                 max_tokens: int) -> LLMResponse: ...


class MLXBackend:
    """Local Qwen via mlx_lm (load once, apply the chat template, generate).

    Accept an optional LoRA adapter path so Phase 6 adapters plug straight in.
    """

    name = "mlx"

    def __init__(self, repo: str, adapter_path: str | None = None):
        self.repo, self.adapter_path = repo, adapter_path

    def complete(self, system, messages, *, temperature, max_tokens) -> LLMResponse:
        raise NotImplementedError


class LLM:
    def __init__(self, backend: Backend, run: RunDir | None = None, role: str = "agent",
                 cache_dir: Path = REPO / ".cache" / "llm", max_usd: float = 5.0):
        self.backend, self.run, self.role = backend, run, role
        self.cache_dir, self.max_usd, self.spent = cache_dir, max_usd, 0.0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, system: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
        blob = json.dumps([self.backend.name, getattr(self.backend, "model", getattr(self.backend, "repo", "")),
                           getattr(self.backend, "adapter_path", None), system, messages,
                           temperature, max_tokens], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def complete(self, system: str, messages: list[dict], *, temperature: float = 0.0,
                 max_tokens: int = 1500, use_cache: bool = True) -> LLMResponse:
        key = self._key(system, messages, temperature, max_tokens)
        path = self.cache_dir / f"{key}.json"
        if use_cache and path.exists():
            resp = LLMResponse(**json.loads(path.read_text()))
            resp.cached = True
        else:
            if self.spent >= self.max_usd:
                raise RuntimeError(f"LLM budget ${self.max_usd} reached for this run")
            t0 = time.time()
            resp = self.backend.complete(system, messages, temperature=temperature, max_tokens=max_tokens)
            resp.latency_s = time.time() - t0
            self.spent += resp.cost_usd
            path.write_text(json.dumps(asdict(resp)))
        if self.run is not None:
            self.run.append_jsonl("llm_calls.jsonl", {
                "role": self.role, "backend": self.backend.name, "key": key, "system": system,
                "messages": messages, "response": resp.text, "cached": resp.cached,
                "tokens": [resp.input_tokens, resp.output_tokens], "cost_usd": resp.cost_usd,
            })
        return resp


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of model output (handles ```json fences). Raises ValueError."""
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        t = next((p[4:] if p.startswith("json") else p for p in parts[1::2]), t)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model output")
    return json.loads(t[start : end + 1])
