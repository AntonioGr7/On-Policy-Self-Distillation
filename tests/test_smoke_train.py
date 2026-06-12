"""End-to-end smoke train. Gated behind SDFT_RUN_SMOKE=1 because it downloads a
small model and needs more compute than a unit test.

    SDFT_RUN_SMOKE=1 pytest tests/test_smoke_train.py -s
"""

import os
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    os.environ.get("SDFT_RUN_SMOKE") != "1",
    reason="set SDFT_RUN_SMOKE=1 to run the end-to-end smoke train (downloads a model)",
)


def test_smoke_train_runs_and_loss_is_finite(tmp_path):
    import torch

    from sdft.cli import build_trainer
    from sdft.config import load_config

    run = load_config(_REPO / "configs" / "smoke_qwen0.5b.yaml")
    run.sdft.output_dir = str(tmp_path / "out")
    run.sdft.max_steps = 2

    trainer = build_trainer(run)
    out = trainer.train()
    assert out.training_loss is not None
    assert torch.isfinite(torch.tensor(out.training_loss))
