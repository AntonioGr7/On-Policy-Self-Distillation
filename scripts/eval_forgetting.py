#!/usr/bin/env python
"""Measure retained general knowledge via the EleutherAI lm-eval-harness.

This is the catastrophic-forgetting probe: run a fixed general-knowledge suite
on a checkpoint and report accuracy. Compare a checkpoint against the base model
(or across continual-learning stages) to see how much internal knowledge is
preserved.

    python scripts/eval_forgetting.py \
        --model outputs/qwen2.5-3b-science \
        --tasks arc_easy,hellaswag,mmlu \
        --limit 200 --output results/forgetting_stage1.json

Requires the eval extra:  uv pip install -e ".[eval]"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_lm_eval(model_path: str, tasks: list[str], limit: int | None, batch_size: str):
    try:
        from lm_eval import simple_evaluate
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "lm-eval is not installed. Install it with: uv pip install -e '.[eval]'"
        ) from exc

    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype=bfloat16",
        tasks=tasks,
        limit=limit,
        batch_size=batch_size,
    )
    return results


def _summarize(results: dict) -> dict:
    """Pull the headline metric per task into a flat {task: score} dict."""
    summary = {}
    for task, metrics in results.get("results", {}).items():
        # prefer acc_norm, then acc, then the first numeric metric
        for key in ("acc_norm,none", "acc,none", "acc_norm", "acc"):
            if key in metrics:
                summary[task] = metrics[key]
                break
        else:
            nums = [v for v in metrics.values() if isinstance(v, (int, float))]
            if nums:
                summary[task] = nums[0]
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Forgetting probe via lm-eval-harness.")
    p.add_argument("--model", required=True, help="Path or HF id of the checkpoint.")
    p.add_argument("--tasks", default="arc_easy,hellaswag", help="Comma-separated task names.")
    p.add_argument("--limit", type=int, default=None, help="Max examples per task (for speed).")
    p.add_argument("--batch_size", default="auto")
    p.add_argument("--output", default=None, help="Write summary JSON here.")
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results = run_lm_eval(args.model, tasks, args.limit, args.batch_size)
    summary = _summarize(results)
    print(json.dumps(summary, indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model": args.model, "summary": summary}, indent=2))
        print(f"[sdft] wrote {out}")


if __name__ == "__main__":
    main()
