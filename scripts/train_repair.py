"""Phase 6: fine-tune the 2B repair model with mlx-lm-lora, SFT then GRPO.

  uv run python scripts/train_repair.py sft  --logs runs/*_gen_repairs/edits.jsonl
  uv run python scripts/train_repair.py grpo --logs runs/*_gen_repairs/edits.jsonl \\
      --resume runs/<sft run>/adapter/adapters.safetensors

Builds the data into the RunDir, then calls mlx-lm-lora with gap/train/configs/<mode>_v1.yaml.
The adapter lands in <run>/adapter; pass it to run_loop.py --adapter.

Two additions for GRPO: the simulator reward is registered in-process, and GRPO prompts are
rendered with enable_thinking=False. mlx-lm-lora's GRPO dataset uses the chat template's
default, which turns Qwen3.5's thinking on and would not match the run-time prompt.
"""

from __future__ import annotations

import argparse
import functools
import sys
from importlib.metadata import version
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gap.train.build_sft_data import build, build_grpo
from gap.train.rewards import score_edit
from gap.utils.runs import RunDir

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "gap" / "train" / "configs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sft", "grpo"])
    ap.add_argument("--logs", nargs="+", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--resume", default=None, help="adapter to start from (GRPO: the SFT one)")
    args = ap.parse_args()

    model = yaml.safe_load((REPO / "configs" / "models.yaml").read_text())["local"]["mlx_repo"]
    cfg = yaml.safe_load((args.config or CONFIGS / f"{args.mode}_v1.yaml").read_text())
    run = RunDir(f"train_{args.mode}", {"logs": [str(p) for p in args.logs], "train": cfg,
                                        "model": model, "resume": args.resume,
                                        "mlx_lm_lora": version("mlx-lm-lora")})
    data = run.path / "data"
    counts = build(args.logs, data) if args.mode == "sft" else build_grpo(args.logs, data)
    print("examples:", counts, flush=True)
    cfg |= {"model": model, "train": True, "train_mode": args.mode, "data": str(data),
            "adapter_path": str(run.path / "adapter"), "resume_adapter_file": args.resume}
    if args.mode == "grpo":
        _setup_grpo(data / "problems")
        cfg["reward_functions"] = "sim_edit_reward"
    from mlx_lm_lora import train
    train.main(cfg)
    run.metrics(mode=args.mode, examples=counts, adapter=str(run.path / "adapter"))


def _setup_grpo(problems_dir: Path) -> None:
    from mlx_lm_lora.trainer import datasets
    from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function

    score = functools.partial(score_edit, problems_dir=problems_dir)

    @register_reward_function("sim_edit_reward")
    def sim_edit_reward(prompts, completions, answer, types=None):
        return [score(a, c) for a, c in zip(answer, completions, strict=True)]

    original = datasets.GRPODataset.__init__

    def init_without_thinking(self, data, tokenizer, *a, **k):
        original(self, data, _NoThinking(tokenizer), *a, **k)

    datasets.GRPODataset.__init__ = init_without_thinking


class _NoThinking:
    """Tokenizer proxy whose chat template always renders with enable_thinking=False."""

    def __init__(self, tok):
        self._tok = tok

    def apply_chat_template(self, *a, **k):
        return self._tok.apply_chat_template(*a, **{**k, "enable_thinking": False})

    def __getattr__(self, name):
        return getattr(self._tok, name)


if __name__ == "__main__":
    main()
