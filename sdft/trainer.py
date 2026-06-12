"""SDFTTrainer: on-policy self-distillation on top of ``transformers.Trainer``.

Per micro-batch:

1. Render the student prompt (prompt only) and the teacher prompt (same prompt
   conditioned on the gold demonstration in-context).
2. Sample a completion on-policy from the student (``lmbda`` controls the chance
   of using the gold demonstration as the completion instead — off-policy).
3. Forward the student over ``[student_prompt | completion]`` (grad on) and the
   teacher over ``[teacher_prompt | completion]`` (grad off). The completion
   token ids are identical in both; only the conditioning context differs.
4. Slice each model's logits to the completion positions and apply the
   generalized JSD / KL loss (``sdft.losses.generalized_jsd_loss``).

The teacher shares the student's *initial weights* (full fine-tuning,
paper-faithful) and is steered purely by the in-context demonstration; an
optional EMA callback keeps it tracking the student.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import Trainer

from sdft.config import DataConfig, SDFTConfig
from sdft.data import DEMO_KEY, PROMPT_KEY
from sdft.generation import generate_on_policy
from sdft.losses import generalized_jsd_loss
from sdft.models import build_teacher_messages


@dataclass
class SDFTCollator:
    """Render + tokenize student/teacher prompts and the gold completion.

    Prompts are left-padded (so generation and the completion slice line up at a
    uniform prompt length); the gold completion is right-padded.
    """

    tokenizer: object
    model_cfg: object
    max_prompt_length: int = 512
    max_completion_length: int = 256

    def _encode_prompts(self, texts: list[str]):
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
            add_special_tokens=False,  # chat template already added them
        )
        return enc["input_ids"], enc["attention_mask"]

    def __call__(self, features: list[dict]) -> dict:
        tok = self.tokenizer

        student_texts, teacher_texts, demos = [], [], []
        for ex in features:
            msgs = ex[PROMPT_KEY]
            demo = ex[DEMO_KEY]
            student_texts.append(
                tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            )
            t_msgs = build_teacher_messages(msgs, demo, self.model_cfg)
            teacher_texts.append(
                tok.apply_chat_template(t_msgs, add_generation_prompt=True, tokenize=False)
            )
            demos.append(demo)

        s_ids, s_mask = self._encode_prompts(student_texts)
        t_ids, t_mask = self._encode_prompts(teacher_texts)

        # Gold completion ids (for the off-policy branch), right-padded.
        tok.padding_side = "right"
        demo_enc = tok(
            [d + (tok.eos_token or "") for d in demos],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_completion_length,
            add_special_tokens=False,
        )

        return {
            "prompt_input_ids": s_ids,
            "prompt_attention_mask": s_mask,
            "teacher_prompt_input_ids": t_ids,
            "teacher_prompt_attention_mask": t_mask,
            "demo_completion_ids": demo_enc["input_ids"],
            "demo_completion_mask": demo_enc["attention_mask"],
        }


class SDFTTrainer(Trainer):
    def __init__(
        self,
        *args,
        teacher_model=None,
        sdft_cfg: SDFTConfig | None = None,
        data_cfg: DataConfig | None = None,
        model_name: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sdft_cfg = sdft_cfg or SDFTConfig()
        self.data_cfg = data_cfg or DataConfig()
        self.teacher_model = teacher_model
        self.model_name = model_name or getattr(self.model.config, "_name_or_path", None)
        self._vllm = None

        # Quantized teachers are already placed on-device by bitsandbytes; only
        # move a non-quantized teacher.
        if self.teacher_model is not None:
            if not getattr(self.teacher_model, "is_quantized", False):
                self.teacher_model.to(self.args.device)
            self.teacher_model.eval()
            self.teacher_model.requires_grad_(False)

        # Resolve the Liger fused-linear JSD path, with a graceful fallback.
        self._use_liger = False
        if self.sdft_cfg.use_liger_jsd:
            from sdft.losses import liger_available

            if liger_available():
                self._use_liger = True
            else:
                import warnings

                warnings.warn(
                    "use_liger_jsd=True but liger-kernel is not installed; "
                    "falling back to the dense JSD loss. Install with '.[train]'."
                )

    # -- helpers ---------------------------------------------------------- #
    @staticmethod
    def _completion_slice(prompt_len, comp_len):
        # logit/hidden at position i predicts token i+1; completion token j sits
        # at position prompt_len + j, predicted by position prompt_len + j - 1.
        start = prompt_len - 1
        return slice(start, start + comp_len)

    def _completion_logits(self, model, prompt_ids, prompt_mask, completion_ids, completion_mask):
        """Forward ``[prompt | completion]`` and return logits over completion
        positions, shape ``(B, C, V)``."""
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attn = torch.cat([prompt_mask, completion_mask], dim=1)
        out = model(input_ids=input_ids, attention_mask=attn)
        sl = SDFTTrainer._completion_slice(prompt_ids.shape[1], completion_ids.shape[1])
        return out.logits[:, sl, :]

    def _completion_hidden(self, model, prompt_ids, prompt_mask, completion_ids, completion_mask):
        """Like :meth:`_completion_logits` but returns final hidden states over
        the completion positions, shape ``(B, C, H)`` (for the Liger path)."""
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attn = torch.cat([prompt_mask, completion_mask], dim=1)
        out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        sl = SDFTTrainer._completion_slice(prompt_ids.shape[1], completion_ids.shape[1])
        return out.hidden_states[-1][:, sl, :]

    # -- the SDFT objective ---------------------------------------------- #
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        cfg = self.sdft_cfg
        prompt_ids = inputs["prompt_input_ids"]
        prompt_mask = inputs["prompt_attention_mask"]

        # 1) decide on-policy vs off-policy for this micro-batch
        on_policy = float(torch.rand(1).item()) < cfg.lmbda

        if on_policy:
            completion_ids, completion_mask = self._sample_completion(model, prompt_ids, prompt_mask)
        else:
            completion_ids = inputs["demo_completion_ids"]
            completion_mask = inputs["demo_completion_mask"]

        t_prompt_ids = inputs["teacher_prompt_input_ids"]
        t_prompt_mask = inputs["teacher_prompt_attention_mask"]

        if self._use_liger:
            loss = self._liger_loss(
                model, prompt_ids, prompt_mask, t_prompt_ids, t_prompt_mask,
                completion_ids, completion_mask,
            )
        else:
            loss = self._dense_loss(
                model, prompt_ids, prompt_mask, t_prompt_ids, t_prompt_mask,
                completion_ids, completion_mask,
            )

        if return_outputs:
            return loss, {}
        return loss

    # -- completion sampling (on-policy) --------------------------------- #
    def _sample_completion(self, model, prompt_ids, prompt_mask):
        cfg = self.sdft_cfg
        if cfg.generation_backend == "vllm":
            return self._vllm_sample(prompt_ids, prompt_mask)
        unwrapped = self.accelerator.unwrap_model(model)
        sequences, full_mask = generate_on_policy(
            unwrapped,
            self.processing_class,
            prompt_ids,
            prompt_mask,
            max_new_tokens=self.data_cfg.max_completion_length,
            temperature=cfg.gen_temperature,
            top_p=cfg.gen_top_p,
            top_k=cfg.gen_top_k,
            cache_implementation=cfg.cache_implementation,
        )
        p_len = prompt_ids.shape[1]
        return sequences[:, p_len:], full_mask[:, p_len:]

    def _vllm_sample(self, prompt_ids, prompt_mask):
        from sdft.generation import VLLMGenerator

        if self._vllm is None:
            self._vllm = VLLMGenerator(
                self.model_name,
                gpu_memory_utilization=self.sdft_cfg.vllm_gpu_memory_utilization,
                enable_sleep_mode=self.sdft_cfg.vllm_enable_sleep_mode,
            )
        # keep vLLM weights in sync with the (changing) student
        self._vllm.sync_weights(self.accelerator.unwrap_model(self.model))
        return self._vllm.generate(
            prompt_ids,
            prompt_mask,
            self.processing_class,
            max_new_tokens=self.data_cfg.max_completion_length,
            temperature=self.sdft_cfg.gen_temperature,
            top_p=self.sdft_cfg.gen_top_p,
            top_k=self.sdft_cfg.gen_top_k,
        )

    # -- loss variants --------------------------------------------------- #
    def _dense_loss(self, model, p_ids, p_mask, t_ids, t_mask, c_ids, c_mask):
        cfg = self.sdft_cfg
        student_logits = self._completion_logits(model, p_ids, p_mask, c_ids, c_mask)
        with torch.no_grad():
            teacher_logits = self._completion_logits(self.teacher_model, t_ids, t_mask, c_ids, c_mask)
        return generalized_jsd_loss(
            student_logits,
            teacher_logits.to(student_logits.dtype),
            mask=c_mask,
            alpha=cfg.alpha,
            temperature=cfg.temperature,
            top_k=cfg.loss_top_k,
        )

    def _liger_loss(self, model, p_ids, p_mask, t_ids, t_mask, c_ids, c_mask):
        """Fused-linear JSD: never materializes full-vocab logits."""
        from sdft.losses import fused_linear_jsd_loss

        cfg = self.sdft_cfg
        student_hidden = self._completion_hidden(model, p_ids, p_mask, c_ids, c_mask)  # (B,C,H)
        with torch.no_grad():
            teacher_hidden = self._completion_hidden(self.teacher_model, t_ids, t_mask, c_ids, c_mask)

        # flatten to (N, H) over valid completion tokens only
        flat_mask = c_mask.reshape(-1).bool()
        H = student_hidden.shape[-1]
        s_flat = student_hidden.reshape(-1, H)[flat_mask]
        t_flat = teacher_hidden.reshape(-1, H)[flat_mask].to(s_flat.dtype)

        s_w = self.accelerator.unwrap_model(model).get_output_embeddings().weight
        t_w = self.teacher_model.get_output_embeddings().weight.to(s_flat.dtype)
        return fused_linear_jsd_loss(
            s_flat, s_w, t_flat, t_w, alpha=cfg.alpha, temperature=cfg.temperature
        )
