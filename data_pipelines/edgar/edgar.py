"""Pure logic for turning SEC EDGAR XBRL company-facts into SDFT-format QA rows.

This module is intentionally standalone — it does NOT import `sdft`. Its only
contract with the trainer is the output JSONL schema:

    train row:  {"prompt_messages": [...], "demonstration": "<gold answer>"}
    test  row:  {"prompt_messages": [...], "reference": "<answer to match>", ...}

Why XBRL company-facts (not 10-Q HTML)? Facts arrive pre-structured as
(concept, period, value, unit, filing date), so ground truth is exact and dated
— no HTML parsing, no LLM-generated QA, no hallucinated answers. Filtering to
filings *after* a model's knowledge cutoff yields genuinely unseen facts: the
clean test of knowledge injection without forgetting.

All functions here are deterministic and offline-testable; the network/CLI layer
lives in build.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

# --------------------------------------------------------------------------- #
# Metric registry: which us-gaap concepts to extract, how to phrase them, and
# how to format their values. `tags` lists candidate XBRL concept names in
# priority order (companies tag the same idea differently). `kind` drives value
# formatting and unit selection.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Metric:
    key: str
    tags: tuple[str, ...]
    label: str
    kind: str  # "usd" | "usd_per_share" | "shares" | "pure"


METRICS: tuple[Metric, ...] = (
    Metric("revenue",
           ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "RevenuesNet"),
           "total revenue", "usd"),
    Metric("net_income", ("NetIncomeLoss",), "net income", "usd"),
    Metric("operating_income", ("OperatingIncomeLoss",), "operating income", "usd"),
    Metric("gross_profit", ("GrossProfit",), "gross profit", "usd"),
    Metric("rd_expense", ("ResearchAndDevelopmentExpense",),
           "research and development expense", "usd"),
    Metric("eps_diluted", ("EarningsPerShareDiluted",), "diluted earnings per share",
           "usd_per_share"),
    Metric("total_assets", ("Assets",), "total assets", "usd"),
    Metric("cash", ("CashAndCashEquivalentsAtCarryingValue",),
           "cash and cash equivalents", "usd"),
)

_UNIT_FOR_KIND = {"usd": "USD", "usd_per_share": "USD/shares", "shares": "shares"}

_QUARTER_WORD = {"Q1": "first", "Q2": "second", "Q3": "third", "Q4": "fourth"}


@dataclass(frozen=True)
class Fact:
    company: str
    metric_key: str
    label: str
    kind: str
    value: float
    unit: str
    period_end: str       # YYYY-MM-DD
    fiscal_year: int | None
    fiscal_period: str | None  # "FY" | "Q1".."Q4"
    form: str             # "10-K" | "10-Q" | ...
    filed: str            # YYYY-MM-DD
    accession: str

    @property
    def dedup_key(self) -> tuple:
        return (self.metric_key, self.fiscal_year, self.fiscal_period, self.period_end)


# --------------------------------------------------------------------------- #
# Formatting / phrasing
# --------------------------------------------------------------------------- #
def format_value(value: float, kind: str) -> str:
    if kind == "usd_per_share":
        return f"${value:.2f} per share"
    if kind == "shares":
        return f"{value / 1e6:.1f} million shares"
    if kind == "pure":
        return f"{value:,.0f}"
    # usd
    sign = "-" if value < 0 else ""
    v = abs(value)
    if v >= 1e9:
        return f"{sign}${v / 1e9:.2f} billion"
    if v >= 1e6:
        return f"{sign}${v / 1e6:.1f} million"
    return f"{sign}${v:,.0f}"


def period_label(fact: Fact) -> str:
    if fact.fiscal_period == "FY" or fact.fiscal_period is None:
        return f"fiscal year {fact.fiscal_year}"
    word = _QUARTER_WORD.get(fact.fiscal_period, fact.fiscal_period)
    return f"the {word} quarter of fiscal {fact.fiscal_year}"


def _parse(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _duration_ok(entry: dict, fiscal_period: str | None) -> bool:
    """Keep only the clean single-period value, not YTD cumulatives.

    Flow concepts (revenue, income) appear in a 10-Q as both a ~3-month value and
    a year-to-date value sharing the same `end`. We keep ~quarter-length spans for
    quarters and ~year-length spans for FY. Instant concepts (no `start`) pass.
    """
    start = entry.get("start")
    end = entry.get("end")
    if not start or not end:
        return True  # instant (balance-sheet) fact
    days = (_parse(end) - _parse(start)).days
    if fiscal_period == "FY":
        return 350 <= days <= 380
    return 80 <= days <= 100  # a single quarter


def extract_facts(
    companyfacts: dict,
    since: date,
    *,
    metrics: tuple[Metric, ...] = METRICS,
    forms: tuple[str, ...] = ("10-Q", "10-K"),
) -> list[Fact]:
    """Pull dated facts filed on/after `since` from an XBRL company-facts dict.

    De-duplicates per (metric, fiscal year, fiscal period, period end), keeping
    the most recently filed value (handles amended/restated filings).
    """
    company = companyfacts.get("entityName", "the company")
    us_gaap = (companyfacts.get("facts") or {}).get("us-gaap") or {}

    best: dict[tuple, Fact] = {}
    for metric in metrics:
        concept = next((us_gaap[t] for t in metric.tags if t in us_gaap), None)
        if concept is None:
            continue
        unit_key = _UNIT_FOR_KIND.get(metric.kind)
        units = concept.get("units") or {}
        entries = units.get(unit_key) if unit_key else next(iter(units.values()), [])
        for e in entries or []:
            if e.get("form") not in forms:
                continue
            filed = e.get("filed")
            if not filed or _parse(filed) < since:
                continue
            if not _duration_ok(e, e.get("fp")):
                continue
            fact = Fact(
                company=company,
                metric_key=metric.key,
                label=metric.label,
                kind=metric.kind,
                value=float(e["val"]),
                unit=unit_key or "",
                period_end=e.get("end", ""),
                fiscal_year=e.get("fy"),
                fiscal_period=e.get("fp"),
                form=e["form"],
                filed=filed,
                accession=e.get("accn", ""),
            )
            prev = best.get(fact.dedup_key)
            if prev is None or _parse(filed) > _parse(prev.filed):
                best[fact.dedup_key] = fact

    return sorted(best.values(), key=lambda f: (f.metric_key, f.period_end))


# --------------------------------------------------------------------------- #
# QA row builders  (output schema is the SDFT/json contract)
#
# Paraphrase augmentation matters more than fact count for knowledge injection:
# a fact seen in one surface form rarely generalizes. We emit N varied question
# phrasings per fact for training, and reserve a DISTINCT phrasing for the test
# set so closed-book eval measures recall, not string-echo.
# --------------------------------------------------------------------------- #
_QUESTION_TEMPLATES = (
    "According to {company}'s {form} filing, what was {label} for {period}?",
    "What was {company}'s {label} in {period}?",
    "In {period}, how much was {company}'s {label}?",
    "How much {label} did {company} report for {period}?",
    "Report {company}'s {label} for {period}.",
    "For {period}, what {label} did {company} report?",
)
# Held out from training — used only for the test set.
_TEST_QUESTION = "{company} reported how much {label} for {period}?"

_ANSWER_TEMPLATES = (
    "{company}'s {label} for {period} was {value}, as reported in its {form} filed on {filed}.",
    "For {period}, {company} reported {label} of {value} ({form}, filed {filed}).",
    "{company} reported {label} of {value} in {period}.",
)


def _ctx(fact: Fact) -> dict:
    return {
        "company": fact.company,
        "label": fact.label,
        "period": period_label(fact),
        "form": fact.form,
        "filed": fact.filed,
        "value": format_value(fact.value, fact.kind),
    }


def fact_to_train_rows(fact: Fact, n_paraphrases: int = 1) -> list[dict]:
    """Emit `n_paraphrases` training rows for a fact, each with a different
    question (and rotating answer) phrasing. Templates cycle if N exceeds the
    pool. Deterministic — no randomness, so dataset builds are reproducible."""
    ctx = _ctx(fact)
    n = max(1, n_paraphrases)
    rows = []
    for i in range(n):
        prompt = _QUESTION_TEMPLATES[i % len(_QUESTION_TEMPLATES)].format(**ctx)
        demonstration = _ANSWER_TEMPLATES[i % len(_ANSWER_TEMPLATES)].format(**ctx)
        rows.append({"prompt_messages": [{"role": "user", "content": prompt}],
                     "demonstration": demonstration})
    return rows


def fact_to_train_row(fact: Fact) -> dict:
    """Single training row (first phrasing). Convenience over fact_to_train_rows."""
    return fact_to_train_rows(fact, 1)[0]


def fact_to_test_row(fact: Fact) -> dict:
    """Closed-book probe: the HELD-OUT question phrasing (never seen in training)
    and the value to match. Keeps provenance fields for analysis."""
    ctx = _ctx(fact)
    return {
        "prompt_messages": [{"role": "user", "content": _TEST_QUESTION.format(**ctx)}],
        "reference": ctx["value"],
        "metric": fact.metric_key,
        "period_end": fact.period_end,
        "form": fact.form,
        "filed": fact.filed,
    }


# --------------------------------------------------------------------------- #
# NASDAQ universe parsing (the fetch lives in build.py)
# --------------------------------------------------------------------------- #
def parse_nasdaq_symbols(listed_txt: str) -> list[str]:
    """Parse nasdaqtrader's pipe-delimited ``nasdaqlisted.txt`` into common-stock
    symbols, excluding test issues, ETFs, and the trailing footer line.

    Header: Symbol|Security Name|Market Category|Test Issue|Financial Status|
            Round Lot Size|ETF|NextShares
    """
    lines = [ln for ln in listed_txt.splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0].split("|")
    col = {name: i for i, name in enumerate(header)}
    out = []
    for line in lines[1:]:
        if line.startswith("File Creation Time"):
            continue
        cols = line.split("|")
        if len(cols) < len(header):
            continue
        if cols[col.get("Test Issue", 3)].strip() == "Y":
            continue
        if "ETF" in col and cols[col["ETF"]].strip() == "Y":
            continue
        sym = cols[col.get("Symbol", 0)].strip()
        if sym:
            out.append(sym)
    return out
