"""Configuration dataclasses for SDFT, loadable from YAML.

A run is described by a single :class:`RunConfig` which nests
:class:`ModelConfig`, :class:`DataConfig` and :class:`SDFTConfig`. YAML files
may also declare a ``family`` include (``configs/families/<name>.yaml``) whose
fields are merged in as defaults under ``model`` — this is what lets you switch
model family with a single line.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Directory holding configs/ — resolved relative to the repo root so that
# `family:` includes can be located regardless of the working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_FAMILIES_DIR = _REPO_ROOT / "configs" / "families"


@dataclass
class ModelConfig:
    """Which model to train and how to load it."""

    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    family: str = "qwen"
    # dtype for weights; "bfloat16" on A100, "float32" on CPU dev box.
    dtype: str = "bfloat16"
    attn_implementation: str | None = None  # e.g. "flash_attention_2" on A100
    trust_remote_code: bool = False
    # Teacher prompt template: how the gold demonstration is injected in-context.
    # Provided by the family include; "{demonstration}" is substituted.
    teacher_demo_template: str | None = None
    # Whether the teacher is a separate frozen copy (full-FT, paper-faithful) or
    # shares weights with the student via a forward pass that disables grads.
    separate_teacher: bool = True
    # Quantize the frozen teacher to save memory (it only does forward passes).
    # None | "nf4" (4-bit) | "int8" (8-bit). Requires bitsandbytes + CUDA.
    # NOTE: a quantized teacher cannot be EMA-synced (weights are frozen), so
    # sync_ref_model is auto-disabled when this is set.
    teacher_quantization: str | None = None


@dataclass
class DataConfig:
    """Where training data comes from and how prompts are built."""

    dataset_name: str = "smoke"  # "tooluse" | "science" | "smoke" | "json"
    data_path: str | None = None  # for dataset_name == "json"
    split: str = "train"
    max_samples: int | None = None  # truncate for quick runs
    max_prompt_length: int = 512
    max_completion_length: int = 256
    shuffle: bool = True
    seed: int = 42


@dataclass
class SDFTConfig:
    """The self-distillation objective and the optimization schedule.

    Fields up to ``report_to`` mirror common ``transformers.TrainingArguments``
    and are forwarded when the trainer is built.
    """

    # --- distillation objective ---
    # alpha=0 -> forward KL (student||teacher), the paper default.
    # alpha=1 -> reverse KL. 0<alpha<1 -> generalized JSD.
    alpha: float = 0.0
    temperature: float = 1.0  # softmax temperature for the distillation loss
    # Restrict the KL/JSD to the teacher's top-k tokens per position (denoises and
    # trims the loss tensor). None = full vocab.
    loss_top_k: int | None = None
    # Use Liger-Kernel's fused-linear JSD: avoids materializing full-vocab logits
    # (big memory/speed win at large vocab). Falls back to the dense loss if Liger
    # is not installed. Requires the student/teacher to expose hidden states.
    use_liger_jsd: bool = False
    # on-policy sampling temperature for student generation
    gen_temperature: float = 1.0
    gen_top_p: float = 1.0
    gen_top_k: int | None = None
    # fraction of completions sampled from the student (vs. the teacher's demo
    # trajectory). 1.0 = fully on-policy (paper default for SDFT).
    lmbda: float = 1.0

    # --- EMA self-teacher coupling ---
    sync_ref_model: bool = False
    ref_model_mixup_alpha: float = 0.6  # new_teacher = a*student + (1-a)*teacher
    ref_model_sync_steps: int = 512

    # --- optimization (forwarded to TrainingArguments) ---
    output_dir: str = "outputs/run"
    learning_rate: float = 1e-5
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"  # "adamw_bnb_8bit" on A100 to save optimizer mem
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int | None = 2
    seed: int = 42
    deepspeed: str | None = None  # path to a ds_zero*.json
    report_to: str = "none"  # "wandb" on real runs

    # --- generation backend ---
    generation_backend: str = "transformers"  # "transformers" | "vllm"
    vllm_gpu_memory_utilization: float = 0.3
    vllm_enable_sleep_mode: bool = True  # free vLLM GPU mem between generations
    # torch.compile + static KV cache for the transformers generation path.
    compile_generation: bool = False
    cache_implementation: str | None = None  # e.g. "static" for compiled gen


@dataclass
class RunConfig:
    """Top-level config for a single SDFT run."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sdft: SDFTConfig = field(default_factory=SDFTConfig)
    run_name: str = "sdft-run"


def _dataclass_from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate a dataclass from a dict, ignoring unknown keys (with a warning)."""
    field_names = {f.name for f in dataclasses.fields(cls)}
    known = {k: v for k, v in data.items() if k in field_names}
    unknown = set(data) - field_names
    if unknown:
        import warnings

        warnings.warn(f"{cls.__name__}: ignoring unknown config keys {sorted(unknown)}")
    return cls(**known)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins). Returns a copy."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _load_family_defaults(family: str) -> dict:
    """Load ``configs/families/<family>.yaml`` if present, else {}."""
    path = _FAMILIES_DIR / f"{family}.yaml"
    if not path.exists():
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: str | Path) -> RunConfig:
    """Load a :class:`RunConfig` from YAML.

    Resolution order for ``model`` fields:
      family include defaults  <  values written in the run config file.

    The run YAML's ``model.family`` (or "qwen") selects the family include.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    model_raw = raw.get("model", {}) or {}
    family = model_raw.get("family", ModelConfig.family)
    family_defaults = _load_family_defaults(family)  # may contain a "model" block

    # Family include is treated as defaults; the run file overrides it.
    merged = _deep_merge(family_defaults, raw)
    merged_model = merged.get("model", {}) or {}

    return RunConfig(
        model=_dataclass_from_dict(ModelConfig, merged_model),
        data=_dataclass_from_dict(DataConfig, merged.get("data", {}) or {}),
        sdft=_dataclass_from_dict(SDFTConfig, merged.get("sdft", {}) or {}),
        run_name=merged.get("run_name", RunConfig.run_name),
    )
