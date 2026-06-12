#!/usr/bin/env python
"""Continual-learning study: SDFT a model through a sequence of tasks and track
both new-task acquisition and retention of general knowledge.

Driven by a sequence YAML (see experiments/continual_example.yaml):

    base: configs/qwen2.5-3b_a100-40g.yaml   # shared run config
    forgetting: { tasks: [arc_easy, hellaswag], limit: 200 }
    stages:
      - name: science
        overrides: { data: {dataset_name: science}, sdft: {output_dir: outputs/cl/science} }
        heldout: data/science_data/test.jsonl
      - name: tooluse
        overrides: { data: {dataset_name: tooluse}, sdft: {output_dir: outputs/cl/tooluse} }
        heldout: data/tooluse_data/test.jsonl

For each stage we: train from the previous checkpoint, snapshot, then evaluate
held-out accuracy on every task seen so far (transfer + backward retention) and
the fixed forgetting suite. Results are written to a CSV + markdown table.

    python scripts/train_continual.py --sequence experiments/continual_example.yaml

Evals are best-effort: missing held-out files or the optional lm-eval/eval
extras are skipped with a warning, so the training sweep still completes.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

from sdft.config import _deep_merge, _dataclass_from_dict  # reuse loader internals
from sdft.config import DataConfig, ModelConfig, RunConfig, SDFTConfig, load_config
from sdft.cli import build_trainer

_PY = sys.executable
_SCRIPTS = Path(__file__).resolve().parent


def _build_stage_run(base_path: str, overrides: dict, start_model: str | None) -> RunConfig:
    """Merge stage overrides onto the base config and point at the start checkpoint."""
    with open(base_path) as fh:
        base_raw = yaml.safe_load(fh) or {}
    merged_raw = _deep_merge(base_raw, overrides or {})
    if start_model is not None:
        merged_raw.setdefault("model", {})["name"] = start_model

    # Reuse load_config's family-merge by writing nothing to disk: replicate it.
    from sdft.config import _load_family_defaults

    family = (merged_raw.get("model", {}) or {}).get("family", ModelConfig.family)
    merged = _deep_merge(_load_family_defaults(family), merged_raw)
    return RunConfig(
        model=_dataclass_from_dict(ModelConfig, merged.get("model", {}) or {}),
        data=_dataclass_from_dict(DataConfig, merged.get("data", {}) or {}),
        sdft=_dataclass_from_dict(SDFTConfig, merged.get("sdft", {}) or {}),
        run_name=merged.get("run_name", RunConfig.run_name),
    )


def _eval_task(model_path: str, heldout: str, out_json: str) -> dict | None:
    if not heldout or not Path(heldout).exists():
        print(f"[cl] skip task eval (no held-out file: {heldout})")
        return None
    cmd = [_PY, str(_SCRIPTS / "eval_task.py"), "--model", model_path,
           "--data", heldout, "--output", out_json]
    if subprocess.run(cmd).returncode == 0 and Path(out_json).exists():
        return json.loads(Path(out_json).read_text())
    return None


def _eval_forgetting(model_path: str, tasks: list[str], limit: int | None, out_json: str) -> dict | None:
    if not tasks:
        return None
    cmd = [_PY, str(_SCRIPTS / "eval_forgetting.py"), "--model", model_path,
           "--tasks", ",".join(tasks), "--output", out_json]
    if limit:
        cmd += ["--limit", str(limit)]
    if subprocess.run(cmd).returncode == 0 and Path(out_json).exists():
        return json.loads(Path(out_json).read_text())
    print("[cl] forgetting eval skipped/failed (is the '.[eval]' extra installed?)")
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Run an SDFT continual-learning sequence.")
    p.add_argument("--sequence", required=True, help="Path to a sequence YAML.")
    p.add_argument("--results_dir", default="results/continual")
    p.add_argument("--skip_eval", action="store_true", help="Train only; no evals.")
    args = p.parse_args()

    with open(args.sequence) as fh:
        seq = yaml.safe_load(fh)
    base_path = seq["base"]
    forget_cfg = seq.get("forgetting", {}) or {}
    stages = seq["stages"]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    start_model: str | None = None
    seen = []  # list of (stage_name, heldout_path)
    table_rows = []

    for i, stage in enumerate(stages):
        name = stage["name"]
        print(f"\n=== Stage {i + 1}/{len(stages)}: {name} ===")
        run = _build_stage_run(base_path, stage.get("overrides", {}), start_model)
        trainer = build_trainer(run)
        trainer.train()
        ckpt = run.sdft.output_dir
        trainer.save_model(ckpt)
        trainer.processing_class.save_pretrained(ckpt)
        del trainer  # free GPU before eval subprocesses
        start_model = ckpt
        seen.append((name, stage.get("heldout")))

        if args.skip_eval:
            continue

        row = {"stage": name, "checkpoint": ckpt}
        # transfer + backward retention: eval all tasks seen so far
        for seen_name, heldout in seen:
            res = _eval_task(ckpt, heldout, str(results_dir / f"{name}__task_{seen_name}.json"))
            if res:
                row[f"task::{seen_name}"] = round(res["accuracy"], 4)
        # forgetting probe
        fres = _eval_forgetting(
            ckpt, forget_cfg.get("tasks", []), forget_cfg.get("limit"),
            str(results_dir / f"{name}__forgetting.json"),
        )
        if fres:
            for bench, score in fres["summary"].items():
                row[f"gen::{bench}"] = round(score, 4)
        table_rows.append(row)

    # write CSV + markdown
    if table_rows:
        cols = sorted({k for r in table_rows for k in r})
        cols = ["stage", "checkpoint"] + [c for c in cols if c not in ("stage", "checkpoint")]
        csv_path = results_dir / "summary.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(table_rows)

        md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in table_rows:
            md.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        (results_dir / "summary.md").write_text("\n".join(md) + "\n")
        print(f"\n[cl] wrote {csv_path} and summary.md")
        print("\n".join(md))


if __name__ == "__main__":
    main()
