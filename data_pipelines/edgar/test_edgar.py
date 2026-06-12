"""Offline tests for the EDGAR logic (no network). Run with:

    PYTHONPATH= uv run python -m pytest data_pipelines/edgar -q
"""

from datetime import date

from edgar import (
    extract_facts,
    fact_to_test_row,
    fact_to_train_row,
    fact_to_train_rows,
    format_value,
    parse_nasdaq_symbols,
    period_label,
)

_NASDAQ_SAMPLE = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
TSLA|Tesla, Inc. - Common Stock|Q|N|N|100|N|N
ZTEST|NASDAQ TEST STOCK|G|Y|N|100|N|N
QQQ|Invesco QQQ Trust|G|N|N|100|Y|N
File Creation Time: 0612202616:00|||||||"""

# Synthetic company-facts covering: pre/post cutoff, a YTD-vs-quarter duplicate,
# an amended restatement, an instant (balance-sheet) fact.
_CF = {
    "entityName": "Acme Corp",
    "facts": {
        "us-gaap": {
            "Revenues": {
                "label": "Revenues",
                "units": {
                    "USD": [
                        # pre-cutoff -> excluded
                        {"start": "2024-01-01", "end": "2024-03-31", "val": 100,
                         "fy": 2024, "fp": "Q1", "form": "10-Q", "filed": "2024-05-01", "accn": "a0"},
                        # post-cutoff Q1, original then AMENDED (keep latest=210)
                        {"start": "2025-01-01", "end": "2025-03-31", "val": 200,
                         "fy": 2025, "fp": "Q1", "form": "10-Q", "filed": "2025-05-01", "accn": "a1"},
                        {"start": "2025-01-01", "end": "2025-03-31", "val": 210,
                         "fy": 2025, "fp": "Q1", "form": "10-Q", "filed": "2025-09-01", "accn": "a1b"},
                        # post-cutoff Q2: 3-month (keep=250) + 6-month YTD (drop=450)
                        {"start": "2025-04-01", "end": "2025-06-30", "val": 250,
                         "fy": 2025, "fp": "Q2", "form": "10-Q", "filed": "2025-08-01", "accn": "a2"},
                        {"start": "2025-01-01", "end": "2025-06-30", "val": 450,
                         "fy": 2025, "fp": "Q2", "form": "10-Q", "filed": "2025-08-01", "accn": "a2"},
                    ]
                },
            },
            "NetIncomeLoss": {
                "label": "Net Income (Loss)",
                "units": {"USD": [
                    {"start": "2024-07-01", "end": "2025-06-30", "val": 1_000_000_000,
                     "fy": 2025, "fp": "FY", "form": "10-K", "filed": "2025-08-15", "accn": "k1"},
                ]},
            },
            "Assets": {  # instant / balance-sheet fact (no start)
                "label": "Assets",
                "units": {"USD": [
                    {"end": "2025-06-30", "val": 9_000_000_000,
                     "fy": 2025, "fp": "Q2", "form": "10-Q", "filed": "2025-08-01", "accn": "a2"},
                ]},
            },
        }
    },
}


def _facts():
    return extract_facts(_CF, since=date(2025, 1, 1))


def test_excludes_precutoff_and_keeps_postcutoff():
    facts = _facts()
    assert all(f.filed >= "2025-01-01" for f in facts)
    assert not any(f.value == 100 for f in facts)  # the 2024 Q1 fact is gone


def test_amended_restatement_keeps_latest():
    rev_q1 = [f for f in _facts() if f.metric_key == "revenue" and f.fiscal_period == "Q1"]
    assert len(rev_q1) == 1
    assert rev_q1[0].value == 210  # amended value, not 200


def test_ytd_duplicate_is_dropped():
    rev_q2 = [f for f in _facts() if f.metric_key == "revenue" and f.fiscal_period == "Q2"]
    assert len(rev_q2) == 1
    assert rev_q2[0].value == 250  # the 3-month value, not the 6-month YTD 450


def test_instant_fact_kept():
    assets = [f for f in _facts() if f.metric_key == "total_assets"]
    assert len(assets) == 1 and assets[0].value == 9_000_000_000


def test_train_row_schema():
    fact = next(f for f in _facts() if f.metric_key == "net_income")
    row = fact_to_train_row(fact)
    assert row["prompt_messages"][0]["role"] == "user"
    assert "net income" in row["prompt_messages"][0]["content"]
    assert "$1.00 billion" in row["demonstration"]
    assert "fiscal year 2025" in row["demonstration"]


def test_test_row_schema():
    fact = next(f for f in _facts() if f.metric_key == "revenue" and f.fiscal_period == "Q2")
    row = fact_to_test_row(fact)
    assert "reference" in row and "prompt_messages" in row
    assert row["metric"] == "revenue"
    # closed-book question must NOT echo the demonstration phrasing verbatim
    assert "as reported in" not in row["prompt_messages"][0]["content"]


def test_paraphrase_augmentation():
    fact = next(f for f in _facts() if f.metric_key == "net_income")
    rows = fact_to_train_rows(fact, n_paraphrases=4)
    prompts = [r["prompt_messages"][0]["content"] for r in rows]
    assert len(rows) == 4
    assert len(set(prompts)) == 4  # all four phrasings distinct
    # every phrasing still teaches the same fact (value present in the demo)
    assert all("$1.00 billion" in r["demonstration"] for r in rows)
    # the held-out test phrasing must not appear among the training phrasings
    test_prompt = fact_to_test_row(fact)["prompt_messages"][0]["content"]
    assert test_prompt not in prompts


def test_single_paraphrase_default():
    fact = next(f for f in _facts() if f.metric_key == "net_income")
    assert len(fact_to_train_rows(fact)) == 1
    assert fact_to_train_rows(fact)[0] == fact_to_train_row(fact)


def test_parse_nasdaq_symbols_filters_test_etf_footer():
    syms = parse_nasdaq_symbols(_NASDAQ_SAMPLE)
    assert syms == ["AAPL", "TSLA"]  # ZTEST (test issue), QQQ (ETF), footer dropped


def test_value_formatting():
    assert format_value(35_082_000_000, "usd") == "$35.08 billion"
    assert format_value(-1_200_000_000, "usd") == "-$1.20 billion"
    assert format_value(4_500_000, "usd") == "$4.5 million"
    assert format_value(2.37, "usd_per_share") == "$2.37 per share"


def test_period_label():
    q2 = next(f for f in _facts() if f.fiscal_period == "Q2" and f.metric_key == "revenue")
    fy = next(f for f in _facts() if f.fiscal_period == "FY")
    assert period_label(q2) == "the second quarter of fiscal 2025"
    assert period_label(fy) == "fiscal year 2025"
