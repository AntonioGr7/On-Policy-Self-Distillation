#!/usr/bin/env python
"""Fetch SEC EDGAR XBRL company-facts and emit SDFT-format train/test JSONL.

Standalone — depends only on the stdlib + `requests` (already present via
transformers). Does not import `sdft`.

SEC requires a descriptive User-Agent with contact info and rate-limits to
~10 req/s. Pass --user-agent "Your Name your@email" (or set SEC_USER_AGENT).

Example:
    python data_pipelines/edgar/build.py \
        --tickers NVDA AAPL MSFT \
        --since 2025-01-01 \
        --out-dir data/edgar \
        --user-agent "Jane Doe jane@example.com"

Output:
    data/edgar/train.jsonl   # {prompt_messages, demonstration}  -> SDFT training
    data/edgar/test.jsonl    # {prompt_messages, reference, ...}  -> closed-book eval
    data/edgar/facts.jsonl   # full provenance for every fact (audit trail)

Point a run config at it:
    data: { dataset_name: json, data_path: data/edgar/train.jsonl }
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# import the sibling pure-logic module (its dir is on sys.path when run directly)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from edgar import (  # noqa: E402
    extract_facts,
    fact_to_test_row,
    fact_to_train_rows,
    parse_nasdaq_symbols,
)

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"


def _get_json(url: str, user_agent: str) -> dict:
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _get_text(url: str, user_agent: str) -> str:
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
    resp.raise_for_status()
    return resp.text


def _ticker_to_cik(user_agent: str) -> dict[str, int]:
    data = _get_json(_TICKERS_URL, user_agent)
    return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}


def main() -> None:
    p = argparse.ArgumentParser(description="Build an SDFT dataset from EDGAR XBRL facts.")
    p.add_argument("--tickers", nargs="+", help="Ticker symbols, e.g. NVDA AAPL. If omitted, sample from NASDAQ.")
    p.add_argument("--tickers-file", help="File with one ticker per line (alternative to --tickers).")
    p.add_argument("--sample", type=int, default=30,
                   help="When no tickers are given: how many to randomly sample from the NASDAQ universe.")
    p.add_argument("--seed", type=int, default=0, help="Seed for the NASDAQ sample (reproducible).")
    p.add_argument("--since", default="2025-01-01", help="Keep facts filed on/after this date (YYYY-MM-DD).")
    p.add_argument("--forms", nargs="+", default=["10-Q", "10-K"], help="Filing forms to include.")
    p.add_argument("--out-dir", default="data/edgar")
    p.add_argument("--max-per-company", type=int, default=None, help="Cap facts per company (most recent first).")
    p.add_argument("--paraphrases", type=int, default=1,
                   help="Question phrasings per fact in train.jsonl (knowledge injection wants 3-5).")
    p.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT"),
                   help="SEC-required User-Agent: 'Name email'. Or set SEC_USER_AGENT.")
    args = p.parse_args()

    if not args.user_agent:
        sys.exit("SEC requires a User-Agent. Pass --user-agent 'Name email' or set SEC_USER_AGENT.")

    since = datetime.strptime(args.since, "%Y-%m-%d").date()
    forms = tuple(args.forms)

    print("[edgar] fetching SEC ticker -> CIK map ...")
    cik_map = _ticker_to_cik(args.user_agent)

    # Resolve the ticker list: explicit list/file wins; otherwise sample from the
    # NASDAQ universe, intersected with SEC's map (so every pick is filable).
    tickers = list(args.tickers or [])
    if args.tickers_file:
        tickers += [ln.strip() for ln in Path(args.tickers_file).read_text().splitlines() if ln.strip()]
    if not tickers:
        print("[edgar] no tickers given -> sampling from the NASDAQ universe ...")
        nasdaq = parse_nasdaq_symbols(_get_text(_NASDAQ_URL, args.user_agent))
        universe = [s for s in nasdaq if s.upper() in cik_map]
        rng = random.Random(args.seed)
        tickers = rng.sample(universe, min(args.sample, len(universe)))
        print(f"[edgar] sampled {len(tickers)} of {len(universe)} filable NASDAQ "
              f"tickers (seed={args.seed})")

    all_facts = []
    for tk in tickers:
        tk = tk.upper()
        cik = cik_map.get(tk)
        if cik is None:
            print(f"[edgar]   {tk}: not found in SEC ticker map, skipping")
            continue
        try:
            cf = _get_json(_FACTS_URL.format(cik=cik), args.user_agent)
        except requests.HTTPError as e:
            print(f"[edgar]   {tk}: companyfacts fetch failed ({e}), skipping")
            continue
        facts = extract_facts(cf, since, forms=forms)
        facts.sort(key=lambda f: f.filed, reverse=True)
        if args.max_per_company:
            facts = facts[: args.max_per_company]
        print(f"[edgar]   {tk} (CIK {cik}): {len(facts)} facts since {since}")
        all_facts.extend(facts)
        time.sleep(0.2)  # be polite to SEC

    if not all_facts:
        sys.exit("[edgar] no facts collected — try an earlier --since or different tickers.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_train = 0
    with open(out / "train.jsonl", "w") as ftr, \
         open(out / "test.jsonl", "w") as fte, \
         open(out / "facts.jsonl", "w") as ffa:
        for f in all_facts:
            for row in fact_to_train_rows(f, args.paraphrases):
                ftr.write(json.dumps(row) + "\n")
                n_train += 1
            fte.write(json.dumps(fact_to_test_row(f)) + "\n")
            ffa.write(json.dumps(f.__dict__) + "\n")

    print(f"[edgar] wrote {len(all_facts)} facts -> {n_train} train rows "
          f"(x{args.paraphrases} paraphrases), {len(all_facts)} test rows")
    print(f"[edgar] point a config at it:  data: {{ dataset_name: json, data_path: {out}/train.jsonl }}")


if __name__ == "__main__":
    main()
