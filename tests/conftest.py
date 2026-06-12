"""Shared pytest skip helpers for GPU/optional-dependency tests.

The CPU suite must stay green on the dev box, so anything needing CUDA or a
heavy optional dependency is gated behind these markers and simply skips when
the requirement is absent.
"""

import os

import pytest

try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover
    _HAS_CUDA = False


def _has(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="needs a CUDA GPU")
requires_liger = pytest.mark.skipif(not _has("liger_kernel"), reason="needs liger-kernel")
requires_bnb = pytest.mark.skipif(not _has("bitsandbytes"), reason="needs bitsandbytes")
requires_vllm = pytest.mark.skipif(not _has("vllm"), reason="needs vllm")

# vLLM tests are very heavy (load a model into the engine); opt in explicitly.
run_vllm = pytest.mark.skipif(
    os.environ.get("SDFT_RUN_VLLM") != "1",
    reason="set SDFT_RUN_VLLM=1 to run the heavy vLLM test",
)
