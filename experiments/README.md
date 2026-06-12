# Experiments: the continual-learning study

This directory documents the protocol for the headline claim: **SDFT acquires a
new task while preserving the model's pre-existing knowledge**, and does so
better than ordinary SFT.

## Protocol

1. **Baseline measurement.** Probe the *base* model on a fixed general-knowledge
   suite (the forgetting probe) and on each task's held-out set.
   ```bash
   python scripts/eval_forgetting.py --model Qwen/Qwen2.5-3B-Instruct \
       --tasks arc_easy,hellaswag,mmlu,gsm8k --limit 500 \
       --output results/base_forgetting.json
   ```
2. **Sequential SDFT.** Train through the task sequence, snapshotting and
   evaluating after each stage:
   ```bash
   python scripts/train_continual.py --sequence experiments/continual_example.yaml
   ```
   After each stage the runner records: held-out accuracy on every task seen so
   far (new-task **transfer** + **backward retention**) and the forgetting-suite
   scores. Output: `results/continual/summary.{csv,md}`.
3. **SFT baseline (the comparison).** Re-run the same sequence with an
   SFT-equivalent objective and compare forgetting deltas. To approximate SFT in
   this framework, set the completion to the gold demonstration and disable the
   on-policy/teacher signal:
   ```yaml
   sdft: { lmbda: 0.0, alpha: 1.0 }   # off-policy + match teacher==gold context
   ```
   (For a strict SFT baseline, a plain next-token cross-entropy trainer is the
   cleaner control; the above is the in-framework approximation.)

## What "success" looks like

| metric | SDFT | SFT (control) |
|---|---|---|
| new-task held-out accuracy | ↑ (acquired) | ↑ (acquired) |
| general-knowledge suite Δ vs. base | ≈ 0 (retained) | ↓ (forgotten) |
| backward retention on earlier tasks | high | lower |

The headline figure is the **forgetting delta**: SDFT should stay close to the
base model's general-knowledge scores across stages while SFT degrades.

## Data preparation (reference tasks)

The built-in `science` / `tooluse` loaders expect HF `save_to_disk` data at
`data/<task>_data/train_data/` (and a JSONL `data/<task>_data/test.jsonl` for
held-out scoring). Each train row should provide chat-style `messages` (or a
`prompt`) and the gold assistant answer; the loader normalizes these to the SDFT
schema `{prompt_messages, demonstration}`. See `sdft/data.py`.

For a quick offline dry run of the whole pipeline, the example sequence uses the
synthetic `smoke` dataset — no data prep or download required.

## Forgetting suites

Good general-knowledge benchmarks for the probe (via lm-eval-harness):
`arc_easy`, `arc_challenge`, `hellaswag`, `winogrande`, `mmlu`, `gsm8k`,
`truthfulqa_mc2`. Use `--limit` while iterating, full sets for the final table.
