"""Correctness tests for the generalized JSD / KL distillation loss."""

import torch
import torch.nn.functional as F

from sdft.losses import generalized_jsd_loss


def _manual_forward_kl(s_logits, t_logits):
    ps = F.softmax(s_logits, dim=-1)
    log_ps = F.log_softmax(s_logits, dim=-1)
    log_pt = F.log_softmax(t_logits, dim=-1)
    return (ps * (log_ps - log_pt)).sum(-1).mean()


def _manual_reverse_kl(s_logits, t_logits):
    pt = F.softmax(t_logits, dim=-1)
    log_ps = F.log_softmax(s_logits, dim=-1)
    log_pt = F.log_softmax(t_logits, dim=-1)
    return (pt * (log_pt - log_ps)).sum(-1).mean()


def test_forward_kl_matches_manual():
    torch.manual_seed(0)
    s = torch.randn(2, 3, 7)
    t = torch.randn(2, 3, 7)
    got = generalized_jsd_loss(s, t, alpha=0.0)
    exp = _manual_forward_kl(s, t)
    assert torch.allclose(got, exp, atol=1e-5), (got, exp)


def test_reverse_kl_matches_manual():
    torch.manual_seed(1)
    s = torch.randn(2, 3, 7)
    t = torch.randn(2, 3, 7)
    got = generalized_jsd_loss(s, t, alpha=1.0)
    exp = _manual_reverse_kl(s, t)
    assert torch.allclose(got, exp, atol=1e-5), (got, exp)


def test_identical_distributions_give_zero():
    torch.manual_seed(2)
    logits = torch.randn(4, 5, 11)
    for alpha in (0.0, 0.5, 1.0):
        loss = generalized_jsd_loss(logits.clone(), logits.clone(), alpha=alpha)
        assert torch.allclose(loss, torch.zeros(()), atol=1e-6), (alpha, loss)


def test_jsd_is_nonnegative_and_symmetric_at_half():
    torch.manual_seed(3)
    s = torch.randn(2, 4, 9)
    t = torch.randn(2, 4, 9)
    jsd_st = generalized_jsd_loss(s, t, alpha=0.5)
    jsd_ts = generalized_jsd_loss(t, s, alpha=0.5)
    assert jsd_st >= -1e-6
    # JSD with alpha=0.5 is symmetric in its two arguments.
    assert torch.allclose(jsd_st, jsd_ts, atol=1e-5), (jsd_st, jsd_ts)


def test_mask_selects_tokens():
    torch.manual_seed(4)
    s = torch.randn(1, 4, 6)
    t = torch.randn(1, 4, 6)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    masked = generalized_jsd_loss(s, t, mask=mask, alpha=0.0)
    # Equivalent to computing the loss only on the first two positions.
    ref = generalized_jsd_loss(s[:, :2], t[:, :2], alpha=0.0)
    assert torch.allclose(masked, ref, atol=1e-5), (masked, ref)


def test_empty_mask_is_zero_not_nan():
    s = torch.randn(1, 3, 5)
    t = torch.randn(1, 3, 5)
    mask = torch.zeros(1, 3)
    loss = generalized_jsd_loss(s, t, mask=mask, alpha=0.0)
    assert torch.isfinite(loss) and float(loss) == 0.0


def test_topk_equals_full_when_k_is_vocab():
    torch.manual_seed(5)
    s = torch.randn(2, 3, 8)
    t = torch.randn(2, 3, 8)
    full = generalized_jsd_loss(s, t, alpha=0.0)
    capped = generalized_jsd_loss(s, t, alpha=0.0, top_k=8)
    assert torch.allclose(full, capped, atol=1e-5), (full, capped)


def test_topk_truncation_is_finite_and_nonneg():
    torch.manual_seed(6)
    s = torch.randn(2, 4, 50)
    t = torch.randn(2, 4, 50)
    for alpha in (0.0, 0.5, 1.0):
        loss = generalized_jsd_loss(s, t, alpha=alpha, top_k=5)
        assert torch.isfinite(loss) and float(loss) >= -1e-6


def test_topk_identical_distributions_zero():
    torch.manual_seed(7)
    logits = torch.randn(2, 3, 20)
    loss = generalized_jsd_loss(logits.clone(), logits.clone(), alpha=0.0, top_k=4)
    assert torch.allclose(loss, torch.zeros(()), atol=1e-6)


def test_liger_available_returns_bool():
    from sdft.losses import liger_available

    assert isinstance(liger_available(), bool)
