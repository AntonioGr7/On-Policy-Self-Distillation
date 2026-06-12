#!/usr/bin/env python
"""Task-specific (held-out) accuracy — the transfer / acquisition metric.

Generates answers from a checkpoint on a held-out set and scores them. The
default scorer is a normalized containment match (does the reference answer
appear in the model output, case/space-insensitive); pass --scorer exact for
strict equality. For richer tasks (tool-use JSON, graded science answers),
import this module and provide your own scorer.

    python scripts/eval_task.py \
        --model outputs/qwen2.5-3b-science \
        --data data/science_data/test.jsonl \
        --limit 200 --output results/task_stage1.json

The held-out file is JSONL with rows {prompt|prompt_messages, reference}.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def score_contains(output: str, reference: str) -> float:
    return 1.0 if _normalize(reference) in _normalize(output) else 0.0


def score_exact(output: str, reference: str) -> float:
    return 1.0 if _normalize(output) == _normalize(reference) else 0.0


SCORERS = {"contains": score_contains, "exact": score_exact}


def _load_rows(path: str, limit: int | None):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser(description="Held-out task accuracy for SDFT.")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="JSONL with {prompt, reference} rows.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--scorer", choices=list(SCORERS), default="contains")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    scorer = SCORERS[args.scorer]
    rows = _load_rows(args.data, args.limit)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    total, correct = 0, 0.0
    for row in rows:
        msgs = row.get("prompt_messages") or [{"role": "user", "content": row["prompt"]}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        correct += scorer(gen, row["reference"])
        total += 1

    acc = correct / total if total else 0.0
    summary = {"model": args.model, "data": args.data, "scorer": args.scorer,
               "n": total, "accuracy": acc}
    print(json.dumps(summary, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"[sdft] wrote {out}")


if __name__ == "__main__":
    main()
