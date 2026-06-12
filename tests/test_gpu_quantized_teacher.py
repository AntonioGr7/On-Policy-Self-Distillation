"""GPU validation: nf4-quantized teacher loads, runs, and disables EMA sync.

Skips unless CUDA + bitsandbytes are present. Run on the A100:
    pytest tests/test_gpu_quantized_teacher.py -v
Downloads Qwen2.5-0.5B (small, open).
"""

import warnings

import torch

from conftest import requires_bnb, requires_cuda

from sdft.config import ModelConfig
from sdft.models import build_teacher_messages, load_student_and_teacher

_SMALL = "Qwen/Qwen2.5-0.5B-Instruct"


@requires_cuda
@requires_bnb
def test_quantized_teacher_loads_and_forwards():
    cfg = ModelConfig(name=_SMALL, family="qwen", dtype="bfloat16", teacher_quantization="nf4")
    student, teacher, tok = load_student_and_teacher(cfg)
    assert teacher is not None
    assert getattr(teacher, "is_quantized", False), "teacher should report quantized"

    msgs = build_teacher_messages([{"role": "user", "content": "Hi"}], "Hello!", cfg)
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(teacher.device)
    with torch.no_grad():
        out = teacher(**enc)
    assert torch.isfinite(out.logits).all()


@requires_cuda
@requires_bnb
def test_quantized_teacher_disables_ema_sync():
    """build_trainer must downgrade sync_ref_model=True to False (with a warning)
    when the teacher is quantized, since frozen quant weights can't be blended."""
    from sdft.cli import build_trainer
    from sdft.config import DataConfig, RunConfig, SDFTConfig

    run = RunConfig(
        model=ModelConfig(name=_SMALL, family="qwen", teacher_quantization="nf4"),
        data=DataConfig(dataset_name="smoke", max_samples=4),
        sdft=SDFTConfig(sync_ref_model=True, max_steps=1, output_dir="outputs/_test_q"),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer = build_trainer(run)
        assert any("quantized teacher" in str(w.message) for w in caught)
    # no EMA callback should have been attached
    from sdft.callbacks import EMATeacherSyncCallback

    assert not any(isinstance(cb, EMATeacherSyncCallback) for cb in trainer.callback_handler.callbacks)
