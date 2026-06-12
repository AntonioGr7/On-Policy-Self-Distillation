"""GPU validation: Liger fused-linear JSD must match the dense loss.

This is the numerical check that the memory-efficient path computes the same
objective as the reference dense implementation. Skips unless CUDA + liger are
present. Run on the A100:  pytest tests/test_gpu_liger.py -v
"""

import torch

from conftest import requires_cuda, requires_liger

from sdft.losses import fused_linear_jsd_loss, generalized_jsd_loss


@requires_cuda
@requires_liger
def test_fused_matches_dense_symmetric_jsd():
    """At alpha=0.5 the JSD is symmetric in student/teacher, so this comparison
    is robust to any student/teacher role convention inside Liger — it isolates
    whether the *fusion* is numerically correct."""
    torch.manual_seed(0)
    device = "cuda"
    N, H, V = 32, 64, 256  # N completion tokens, hidden, vocab
    s_hidden = torch.randn(N, H, device=device, dtype=torch.float32)
    t_hidden = torch.randn(N, H, device=device, dtype=torch.float32)
    W = torch.randn(V, H, device=device, dtype=torch.float32)  # shared head

    # dense reference: project to logits, then generalized JSD (mean over tokens)
    s_logits = (s_hidden @ W.t()).unsqueeze(0)  # (1, N, V)
    t_logits = (t_hidden @ W.t()).unsqueeze(0)
    dense = generalized_jsd_loss(s_logits, t_logits, alpha=0.5, temperature=1.0)

    fused = fused_linear_jsd_loss(s_hidden, W, t_hidden, W, alpha=0.5, temperature=1.0)

    # Allow a modest tolerance; if the reduction convention differs this will
    # fail informatively and we adjust the wrapper in sdft/losses.py.
    assert torch.allclose(dense.to(fused.dtype), fused, atol=1e-2, rtol=1e-2), (
        f"dense={float(dense):.5f} fused={float(fused):.5f}"
    )


@requires_cuda
@requires_liger
def test_fused_endpoints_are_finite():
    torch.manual_seed(1)
    device = "cuda"
    N, H, V = 16, 32, 128
    s_hidden = torch.randn(N, H, device=device)
    t_hidden = torch.randn(N, H, device=device)
    W = torch.randn(V, H, device=device)
    for alpha in (0.0, 1.0):
        loss = fused_linear_jsd_loss(s_hidden, W, t_hidden, W, alpha=alpha, temperature=1.0)
        assert torch.isfinite(loss)
