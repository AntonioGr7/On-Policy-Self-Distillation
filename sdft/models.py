"""Model loading and the model-family registry.

The registry lets you switch model family with a single config line. Each
family entry knows: a sensible default model id, and the default *teacher
demonstration template* — the in-context wrapper that turns a gold answer into
the teacher's conditioning. Anything in a family entry can be overridden by the
run config (and by ``configs/families/<family>.yaml``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sdft.config import ModelConfig

# Default in-context template used to condition the teacher on the gold answer.
# It is rendered into the *system* message (see build_teacher_messages) so the
# user turn stays identical between student and teacher. "{demonstration}" is
# substituted with the gold completion.
_DEFAULT_TEACHER_TEMPLATE = (
    "You are answering a user request. A high-quality reference answer is "
    "provided below for your guidance. Produce a response of the same quality, "
    "in your own words.\n\nReference answer:\n{demonstration}"
)


@dataclass(frozen=True)
class FamilySpec:
    default_model: str
    teacher_demo_template: str = _DEFAULT_TEACHER_TEMPLATE
    # Some families (older Gemma/Mistral chat templates) reject a system role;
    # for those we fold the teacher conditioning into the first user turn.
    supports_system_role: bool = True


FAMILY_REGISTRY: dict[str, FamilySpec] = {
    "qwen": FamilySpec(default_model="Qwen/Qwen2.5-0.5B-Instruct"),
    "llama": FamilySpec(default_model="meta-llama/Llama-3.2-1B-Instruct"),
    "gemma": FamilySpec(
        default_model="google/gemma-2-2b-it",
        supports_system_role=False,
    ),
    "mistral": FamilySpec(
        default_model="mistralai/Mistral-7B-Instruct-v0.3",
        supports_system_role=False,
    ),
}


def get_family_spec(family: str) -> FamilySpec:
    if family not in FAMILY_REGISTRY:
        raise ValueError(
            f"Unknown model family '{family}'. Known: {sorted(FAMILY_REGISTRY)}. "
            "Add an entry to FAMILY_REGISTRY (and optionally a configs/families/ file)."
        )
    return FAMILY_REGISTRY[family]


def resolve_teacher_template(cfg: ModelConfig) -> str:
    """The teacher template, preferring the run config over the family default."""
    if cfg.teacher_demo_template:
        return cfg.teacher_demo_template
    return get_family_spec(cfg.family).teacher_demo_template


def build_teacher_messages(
    prompt_messages: list[dict], demonstration: str, cfg: ModelConfig
) -> list[dict]:
    """Return the teacher's chat messages: the student's messages conditioned on
    the gold demonstration.

    For families that support a system role, the demonstration is injected as a
    system message prepended to the conversation. Otherwise it is prepended to
    the first user turn (so the chat template never sees an unsupported role).
    """
    template = resolve_teacher_template(cfg)
    conditioning = template.format(demonstration=demonstration)
    spec = get_family_spec(cfg.family)
    msgs = [dict(m) for m in prompt_messages]

    if spec.supports_system_role and (not msgs or msgs[0]["role"] != "system"):
        return [{"role": "system", "content": conditioning}, *msgs]
    if spec.supports_system_role:  # already has a system turn -> augment it
        msgs[0] = {
            "role": "system",
            "content": conditioning + "\n\n" + msgs[0]["content"],
        }
        return msgs

    # No system role: fold conditioning into the first user message.
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            msgs[i] = {"role": "user", "content": conditioning + "\n\n" + m["content"]}
            return msgs
    return [{"role": "user", "content": conditioning}, *msgs]


def _torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _quantization_config(kind: str, compute_dtype: torch.dtype):
    """Build a BitsAndBytesConfig for the frozen teacher. CUDA-only."""
    from transformers import BitsAndBytesConfig

    if kind == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    if kind == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unknown teacher_quantization '{kind}'. Use 'nf4', 'int8', or null.")


def load_student_and_teacher(cfg: ModelConfig):
    """Load (student, teacher, tokenizer).

    Full fine-tuning, paper-faithful: when ``cfg.separate_teacher`` the teacher
    is a second frozen copy of the model. The teacher is conditioned on the gold
    demonstration at *prompt* time (see build_teacher_messages), so it starts
    from the same weights as the student but is steered by context.

    The teacher may be loaded quantized (``cfg.teacher_quantization``) to save
    memory — it only ever does forward passes, so 4-bit NF4 is near-free in
    quality. Quantized teachers are static (cannot be EMA-synced).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    compute_dtype = _torch_dtype(cfg.dtype)
    load_kwargs: dict = {
        "torch_dtype": compute_dtype,
        "trust_remote_code": cfg.trust_remote_code,
    }
    if cfg.attn_implementation:
        load_kwargs["attn_implementation"] = cfg.attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(cfg.name, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    student = AutoModelForCausalLM.from_pretrained(cfg.name, **load_kwargs)

    teacher = None
    if cfg.separate_teacher:
        teacher_kwargs = dict(load_kwargs)
        if cfg.teacher_quantization:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "teacher_quantization requires a CUDA GPU. Unset it for CPU runs."
                )
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "teacher_quantization needs bitsandbytes; install the train extra: "
                    "uv pip install -e '.[train]'"
                ) from exc
            teacher_kwargs["quantization_config"] = _quantization_config(
                cfg.teacher_quantization, compute_dtype
            )
            # bitsandbytes places the model on GPU itself.
            teacher_kwargs["device_map"] = {"": torch.cuda.current_device()}
        teacher = AutoModelForCausalLM.from_pretrained(cfg.name, **teacher_kwargs)
        teacher.eval()
        teacher.requires_grad_(False)

    return student, teacher, tokenizer
