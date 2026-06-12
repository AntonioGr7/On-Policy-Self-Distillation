"""Offline validation of the SDFT trainer's core math: on-policy generation,
the completion-logit slice, and student/teacher alignment when the two prompts
have *different* lengths (the crux of SDFT). Uses a tiny randomly-initialized
model — no download, runs on CPU in well under a second.
"""

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from sdft.generation import generate_on_policy
from sdft.losses import generalized_jsd_loss
from sdft.trainer import SDFTTrainer


class _Tok:
    pad_token_id = 0
    eos_token_id = 2


def _tiny_model():
    cfg = Qwen2Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        pad_token_id=0, eos_token_id=2, bos_token_id=1,
    )
    return Qwen2ForCausalLM(cfg).eval()


def test_completion_logit_slice_matches_full_forward():
    torch.manual_seed(0)
    model = _tiny_model()
    B, P = 2, 5
    prompt_ids = torch.randint(3, 128, (B, P))
    prompt_mask = torch.ones(B, P, dtype=torch.long)
    comp_ids = torch.randint(3, 128, (B, 4))
    comp_mask = torch.ones(B, 4, dtype=torch.long)

    sliced = SDFTTrainer._completion_logits(None, model, prompt_ids, prompt_mask, comp_ids, comp_mask)
    full = model(
        input_ids=torch.cat([prompt_ids, comp_ids], 1),
        attention_mask=torch.cat([prompt_mask, comp_mask], 1),
    ).logits
    # logits predicting the completion tokens live at [P-1 : P-1+C]
    assert torch.allclose(full[:, P - 1 : P - 1 + comp_ids.shape[1], :], sliced, atol=1e-5)


def test_student_teacher_align_with_different_prompt_lengths():
    torch.manual_seed(1)
    model = _tiny_model()
    B = 2
    comp_ids = torch.randint(3, 128, (B, 6))
    comp_mask = torch.ones(B, 6, dtype=torch.long)

    s_prompt = torch.randint(3, 128, (B, 5))
    t_prompt = torch.randint(3, 128, (B, 9))  # teacher prompt is longer (has the demo)
    s_mask = torch.ones(B, 5, dtype=torch.long)
    t_mask = torch.ones(B, 9, dtype=torch.long)

    sl = SDFTTrainer._completion_logits(None, model, s_prompt, s_mask, comp_ids, comp_mask)
    tl = SDFTTrainer._completion_logits(None, model, t_prompt, t_mask, comp_ids, comp_mask)
    assert sl.shape == tl.shape == (B, 6, 128)

    loss = generalized_jsd_loss(sl, tl, mask=comp_mask, alpha=0.0)
    assert torch.isfinite(loss)


def test_on_policy_generation_shapes():
    torch.manual_seed(2)
    model = _tiny_model()
    B, P = 2, 5
    prompt_ids = torch.randint(3, 128, (B, P))
    prompt_mask = torch.ones(B, P, dtype=torch.long)

    seq, full_mask = generate_on_policy(
        model, _Tok(), prompt_ids, prompt_mask, max_new_tokens=6, temperature=1.0
    )
    assert seq.shape[0] == B and seq.shape[1] <= P + 6
    assert full_mask.shape == seq.shape
    # prompt positions are never marked as completion
    assert int(full_mask[:, :P].sum()) == 0
