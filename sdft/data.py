"""Datasets for SDFT, normalized to a single schema.

Every example is reduced to::

    {"prompt_messages": [{"role": ..., "content": ...}, ...],
     "demonstration":   "<gold completion text>"}

``prompt_messages`` is what the *student* sees. The *teacher* sees the same
messages plus the demonstration injected in-context (see
``sdft.models.build_teacher_messages``). Keeping a single schema is what lets
the trainer stay model- and task-agnostic.

Built-in loaders:
  * ``smoke``   — tiny synthetic set, no download, for dev-box smoke tests.
  * ``json``    — a local .jsonl with {prompt_messages|prompt, completion} rows.
  * ``tooluse`` / ``science`` — the two reference tasks; expects the reference
    data under ``data/<task>_data/`` (HF ``load_from_disk`` format).
"""

from __future__ import annotations

from pathlib import Path

from sdft.config import DataConfig

PROMPT_KEY = "prompt_messages"
DEMO_KEY = "demonstration"

_REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Synthetic smoke dataset — deterministic, no network, runs on the 4GB box.
# --------------------------------------------------------------------------- #
_SMOKE_FACTS = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Convert 10 kilometers to miles.", "10 kilometers is about 6.21 miles."),
    ("Name the largest planet in the solar system.", "The largest planet is Jupiter."),
    ("What gas do plants absorb during photosynthesis?", "Plants absorb carbon dioxide."),
    ("Who wrote the play 'Hamlet'?", "William Shakespeare wrote 'Hamlet'."),
    ("What is 12 multiplied by 8?", "12 multiplied by 8 is 96."),
    ("What is the boiling point of water at sea level in Celsius?", "It is 100 degrees Celsius."),
    ("Translate 'good morning' into Spanish.", "'Good morning' is 'buenos días' in Spanish."),
]


def _smoke_dataset(cfg: DataConfig):
    from datasets import Dataset

    rows = []
    # Repeat the tiny set so there are enough steps to exercise the loop.
    for i in range(8):
        q, a = _SMOKE_FACTS[i % len(_SMOKE_FACTS)]
        rows.append(
            {
                PROMPT_KEY: [{"role": "user", "content": q}],
                DEMO_KEY: a,
            }
        )
    ds = Dataset.from_list(rows)
    return ds


# --------------------------------------------------------------------------- #
# Local JSONL loader.
# --------------------------------------------------------------------------- #
def _json_dataset(cfg: DataConfig):
    from datasets import load_dataset

    if not cfg.data_path:
        raise ValueError("dataset_name='json' requires data.data_path to a .jsonl file")
    raw = load_dataset("json", data_files=cfg.data_path, split="train")

    def _norm(ex):
        if PROMPT_KEY in ex and ex[PROMPT_KEY]:
            messages = ex[PROMPT_KEY]
        elif "prompt" in ex:
            messages = [{"role": "user", "content": ex["prompt"]}]
        else:
            raise ValueError("json rows need a 'prompt' or 'prompt_messages' field")
        demo = ex.get(DEMO_KEY) or ex.get("completion") or ex.get("response")
        if demo is None:
            raise ValueError("json rows need a 'completion'/'response'/'demonstration' field")
        return {PROMPT_KEY: messages, DEMO_KEY: demo}

    return raw.map(_norm, remove_columns=[c for c in raw.column_names if c not in (PROMPT_KEY, DEMO_KEY)])


# --------------------------------------------------------------------------- #
# Reference tasks (tooluse / science). Expect data prepared on disk.
# --------------------------------------------------------------------------- #
def _reference_dataset(cfg: DataConfig, task: str):
    from datasets import load_from_disk

    default_dir = _REPO_ROOT / "data" / f"{task}_data" / "train_data"
    path = Path(cfg.data_path) if cfg.data_path else default_dir
    if not path.exists():
        raise FileNotFoundError(
            f"{task} data not found at {path}. Prepare it (HF save_to_disk format) "
            f"or set data.data_path. See experiments/README.md."
        )
    raw = load_from_disk(str(path))

    def _norm(ex):
        # The reference data stores chat-style turns; we keep the last user turn
        # (and any prior context) as the prompt, and the assistant gold as demo.
        messages = ex.get("messages") or ex.get(PROMPT_KEY)
        if messages is None:
            messages = [{"role": "user", "content": ex["prompt"]}]
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        gold = ex.get(DEMO_KEY) or ex.get("completion")
        if gold is None:
            assistant = [m for m in messages if m["role"] == "assistant"]
            gold = assistant[-1]["content"] if assistant else ""
        return {PROMPT_KEY: prompt_msgs, DEMO_KEY: gold}

    return raw.map(_norm, remove_columns=raw.column_names)


_LOADERS = {
    "smoke": _smoke_dataset,
    "json": _json_dataset,
    "tooluse": lambda c: _reference_dataset(c, "tooluse"),
    "science": lambda c: _reference_dataset(c, "science"),
}


def load_sdft_dataset(cfg: DataConfig):
    """Load and normalize a dataset to the SDFT schema, applying shuffle/truncate."""
    if cfg.dataset_name not in _LOADERS:
        raise ValueError(
            f"Unknown dataset '{cfg.dataset_name}'. Known: {sorted(_LOADERS)}."
        )
    ds = _LOADERS[cfg.dataset_name](cfg)
    if cfg.shuffle:
        ds = ds.shuffle(seed=cfg.seed)
    if cfg.max_samples is not None:
        ds = ds.select(range(min(cfg.max_samples, len(ds))))
    return ds
