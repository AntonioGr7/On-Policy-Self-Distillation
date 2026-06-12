# EDGAR data pipeline (standalone)

Builds an SDFT dataset of **post-cutoff financial facts** from SEC EDGAR XBRL
company-facts. **Independent of the `sdft` package** — it only emits JSONL in the
format the trainer reads, so you can swap it for another dataset builder without
touching the framework.

## Why this design

- **XBRL company-facts, not 10-Q HTML.** Facts come pre-structured as
  `(concept, period, value, unit, filing date)` — exact, dated ground truth with
  no HTML parsing and no LLM-generated QA (so no hallucinated answers).
- **Post-cutoff filter.** Keep only filings on/after `--since`, so the facts are
  genuinely unseen by the base model — the clean test of knowledge injection
  without forgetting.
- **Clean single-period values.** Drops year-to-date cumulatives and keeps the
  amended/restated value when a period was refiled.

## Usage

SEC requires a descriptive User-Agent and rate-limits requests.

**Explicit tickers:**
```bash
python data_pipelines/edgar/build.py \
    --tickers NVDA AAPL MSFT \
    --since 2025-01-01 \
    --paraphrases 4 \
    --out-dir data/edgar \
    --user-agent "Your Name your@email.com"     # or set SEC_USER_AGENT
```

**Or sample the NASDAQ universe** (omit `--tickers`): pulls nasdaqtrader's
listed symbols, keeps those filable on SEC (drops test issues + ETFs), and
randomly samples `--sample N` (reproducible via `--seed`):
```bash
python data_pipelines/edgar/build.py \
    --sample 30 --seed 0 \
    --since 2025-01-01 --paraphrases 4 \
    --out-dir data/edgar \
    --user-agent "Your Name your@email.com"
```

`--paraphrases N` emits N varied question phrasings per fact in `train.jsonl`
(a held-out phrasing is reserved for `test.jsonl`). For knowledge injection,
repetition is the make-or-break lever — use **3–5** here; one phrasing per fact
rarely sticks in weights.

Outputs:
- `data/edgar/train.jsonl` — `{prompt_messages, demonstration}` → SDFT training
- `data/edgar/test.jsonl`  — `{prompt_messages, reference, ...}` → closed-book eval
- `data/edgar/facts.jsonl` — full provenance per fact (audit trail)

Then point a run config at it:
```yaml
data:
  dataset_name: json
  data_path: data/edgar/train.jsonl
```
and evaluate closed-book recall after training:
```bash
python scripts/eval_task.py --model <ckpt> --data data/edgar/test.jsonl --scorer contains
```

## Tests (offline, no network)

```bash
PYTHONPATH= uv run python -m pytest data_pipelines/edgar -q
```

## Caveats / knobs

- **Set `--since` past the model's training cutoff** (Qwen2.5 ≈ 2023, so 2025
  filings are safely unseen). Verify per model.
- **Exact-match eval is brittle for numbers** (rounding/format). The default
  formatting is consistent between train and test; for stricter eval, add a
  numeric-tolerant scorer in `scripts/eval_task.py`.
- **Metric coverage** is the registry in `edgar.py` (`METRICS`); add concepts
  there. Revenue tagging varies by company, so several candidate tags are tried.
- This is a *knowledge-injection* dataset (factual recall). Consider pairing it
  with a grounded-reasoning slice for a second axis of evidence.
