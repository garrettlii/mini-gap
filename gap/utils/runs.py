"""Run folders. OWNER: done

Every experiment writes runs/<timestamp>_<name>/ containing:
  config.json    everything needed to rerun it
  *.jsonl        append-only logs (episodes, llm_calls, edits)
  metrics.json   final numbers; the only file report.py reads
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

RUNS_ROOT = Path(__file__).resolve().parents[2] / "runs"


def to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if hasattr(obj, "model_dump"):  # pydantic
        return obj.model_dump(by_alias=True)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"


class RunDir:
    def __init__(self, name: str, config: dict[str, Any], root: Path = RUNS_ROOT):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = root / f"{stamp}_{name}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.write_json("config.json", {**config, "git_rev": _git_rev(), "name": name})

    def file(self, rel: str) -> Path:
        p = self.path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_json(self, rel: str, obj: Any) -> Path:
        p = self.file(rel)
        p.write_text(json.dumps(to_jsonable(obj), indent=2))
        return p

    def append_jsonl(self, rel: str, obj: Any) -> None:
        with self.file(rel).open("a") as f:
            f.write(json.dumps(to_jsonable(obj)) + "\n")

    def metrics(self, **values: Any) -> Path:
        return self.write_json("metrics.json", values)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
