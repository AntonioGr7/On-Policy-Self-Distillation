"""Distillation losses for SDFT.

The objective is a *generalized Jensen-Shannon divergence* between the student
and teacher next-token distributions, evaluated only on the completion tokens
the student generated on-policy. This is the same family of losses used by
on-policy GKD; the ``alpha`` knob recovers the standard limits:

* ``alpha == 0`` -> forward KL,  ``KL(student || teacher)``  (SDFT paper default)
* ``alpha == 1`` -> reverse KL,  ``KL(teacher || student)``
* ``0 < alpha < 1`` -> generalized JSD with mixture ``m = a*student + (1-a)*teacher``

We follow the convention where the mixture is weighted by ``alpha`` on the
student side, and ``JSD = alpha*KL(teacher||m) + (1-alpha)*KL(student||m)``,
so the limits above hold continuously.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _truncate_to_teacher_topk(student_logits, teacher_logits, top_k):
    """Restrict both logit tensors to the teacher's top-k token ids per position.

    Reduces noise from the long tail and shrinks the loss tensor. The subset is
    re-normalized by the downstream log_softmax, so this is the standard
    top-k knowledge-distillation approximation.
    """
    k = min(top_k, teacher_logits.shape[-1])
    topk_idx = teacher_logits.topk(k, dim=-1).indices  # (B, T, k)
    return student_logits.gather(-1, topk_idx), teacher_logits.gather(-1, topk_idx)


def generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    alpha: float = 0.0,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Per-token generalized JSD, averaged over masked (completion) tokens.

    Args:
        student_logits: ``(B, T, V)`` logits from the trainable student.
        teacher_logits: ``(B, T, V)`` logits from the demonstration-conditioned
            teacher, already aligned to the *same* token positions as the
            student (the caller is responsible for the alignment / shift).
        mask: ``(B, T)`` boolean/float mask selecting completion tokens. If
            ``None`` all positions are used.
        alpha: mixture weight (see module docstring). Clamped to ``[0, 1]``.
        temperature: softmax temperature applied to both logits.

    Returns:
        Scalar loss (mean over masked tokens). Returns ``0`` if the mask is
        empty, so a degenerate micro-batch never produces NaNs.
    """
    if top_k is not None:
        student_logits, teacher_logits = _truncate_to_teacher_topk(
            student_logits, teacher_logits, top_k
        )

    if temperature != 1.0:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

    student_logp = F.log_softmax(student_logits, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits, dim=-1)

    # --- exact limits avoid log(0) at the endpoints ---
    if alpha <= 0.0:
        # forward KL: KL(student || teacher) = sum_x p_s (log p_s - log p_t)
        per_token = F.kl_div(teacher_logp, student_logp, reduction="none", log_target=True).sum(-1)
    elif alpha >= 1.0:
        # reverse KL: KL(teacher || student)
        per_token = F.kl_div(student_logp, teacher_logp, reduction="none", log_target=True).sum(-1)
    else:
        log_a = torch.log(torch.tensor(alpha, device=student_logp.device, dtype=student_logp.dtype))
        log_1ma = torch.log(
            torch.tensor(1.0 - alpha, device=student_logp.device, dtype=student_logp.dtype)
        )
        # log of mixture m = a*student + (1-a)*teacher  (stable via logsumexp)
        mixture_logp = torch.logsumexp(
            torch.stack([student_logp + log_a, teacher_logp + log_1ma], dim=0), dim=0
        )
        kl_teacher = F.kl_div(mixture_logp, teacher_logp, reduction="none", log_target=True).sum(-1)
        kl_student = F.kl_div(mixture_logp, student_logp, reduction="none", log_target=True).sum(-1)
        per_token = alpha * kl_teacher + (1.0 - alpha) * kl_student

    if mask is None:
        return per_token.mean()

    mask = mask.to(per_token.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (per_token * mask).sum() / denom


def liger_available() -> bool:
    try:
        import liger_kernel.transformers  # noqa: F401

        return True
    except Exception:
        return False


def fused_linear_jsd_loss(
    student_hidden: torch.Tensor,
    student_lm_head_weight: torch.Tensor,
    teacher_hidden: torch.Tensor,
    teacher_lm_head_weight: torch.Tensor,
    alpha: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Memory-efficient JSD using Liger-Kernel's fused linear head + JSD.

    Inputs are the *hidden states* at the completion positions (already masked /
    flattened to ``(N, H)``) and the output-embedding weights ``(V, H)``. Liger
    fuses the projection with the JSD so the full-vocab logits are never
    materialized — the key memory win for large-vocab distillation.

    The caller must guarantee Liger is installed (see :func:`liger_available`).
    ``alpha`` maps to Liger's ``jsd_beta`` (student-side mixture weight);
    endpoint behavior (0 -> forward KL, 1 -> reverse KL) follows Liger's
    convention and should be sanity-checked on GPU against the dense loss.
    """
    from liger_kernel.transformers.fused_linear_jsd import LigerFusedLinearJSDLoss

    loss_fn = LigerFusedLinearJSDLoss(jsd_beta=float(alpha), temperature=float(temperature))
    return loss_fn(
        student_hidden,
        student_lm_head_weight,
        teacher_hidden,
        teacher_lm_head_weight,
    )
