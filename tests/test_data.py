"""Schema and config-merge tests (no model download, no GPU)."""

from pathlib import Path

from sdft.config import load_config
from sdft.data import DEMO_KEY, PROMPT_KEY, load_sdft_dataset

_REPO = Path(__file__).resolve().parent.parent


def test_smoke_dataset_schema():
    from sdft.config import DataConfig

    ds = load_sdft_dataset(DataConfig(dataset_name="smoke", shuffle=False))
    assert len(ds) > 0
    ex = ds[0]
    assert PROMPT_KEY in ex and DEMO_KEY in ex
    assert isinstance(ex[PROMPT_KEY], list)
    assert ex[PROMPT_KEY][0]["role"] == "user"
    assert isinstance(ex[DEMO_KEY], str) and ex[DEMO_KEY]


def test_max_samples_truncates():
    from sdft.config import DataConfig

    ds = load_sdft_dataset(DataConfig(dataset_name="smoke", max_samples=3))
    assert len(ds) == 3


def test_load_config_merges_family_defaults():
    run = load_config(_REPO / "configs" / "smoke_qwen0.5b.yaml")
    assert run.model.family == "qwen"
    # teacher_demo_template comes from the qwen family include, not the run file.
    assert run.model.teacher_demo_template is not None
    assert "{demonstration}" in run.model.teacher_demo_template
    assert run.sdft.alpha == 0.0
    assert run.sdft.max_steps == 4


def test_run_config_overrides_family():
    # A run file value should win over the family default.
    run = load_config(_REPO / "configs" / "smoke_qwen0.5b.yaml")
    assert run.model.dtype == "float32"  # set in the run file, overriding bfloat16


def test_optimization_config_fields_parse():
    run = load_config(_REPO / "configs" / "qwen2.5-7b_a100-40g_optimized.yaml")
    assert run.model.teacher_quantization == "nf4"
    assert run.sdft.use_liger_jsd is True
    assert run.sdft.loss_top_k == 256
    assert run.sdft.optim == "paged_adamw_8bit"
    # quantized teacher => EMA sync must not be requested in this config
    assert run.sdft.sync_ref_model is False


def test_teacher_messages_for_system_and_nonsystem_families():
    from sdft.config import ModelConfig
    from sdft.models import build_teacher_messages

    prompt = [{"role": "user", "content": "Q?"}]
    qwen = ModelConfig(family="qwen", teacher_demo_template="REF: {demonstration}")
    tm = build_teacher_messages(prompt, "gold", qwen)
    assert tm[0]["role"] == "system" and "gold" in tm[0]["content"]
    assert tm[-1]["role"] == "user"

    gemma = ModelConfig(family="gemma", teacher_demo_template="REF: {demonstration}")
    tm2 = build_teacher_messages(prompt, "gold", gemma)
    # No system role: conditioning folded into the user turn.
    assert all(m["role"] != "system" for m in tm2)
    assert "gold" in tm2[0]["content"]
