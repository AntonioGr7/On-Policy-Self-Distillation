"""Training callbacks for SDFT.

The EMA self-teacher coupling keeps the teacher close to the student over time:
periodically ``teacher <- alpha*student + (1-alpha)*teacher``. With the teacher
also conditioned on the demonstration in-context, this lets the teaching signal
improve as the student learns, without ever drifting too far from a stable
reference (which would reintroduce forgetting).
"""

from __future__ import annotations

import torch
from transformers import TrainerCallback


class EMATeacherSyncCallback(TrainerCallback):
    """Periodically blend the student's weights into the teacher (EMA).

    Args:
        trainer: the SDFTTrainer (used to reach ``teacher_model`` and the
            unwrapped student).
        alpha: mixup weight for the *student* contribution.
        sync_steps: blend every ``sync_steps`` optimizer steps.
    """

    def __init__(self, trainer, alpha: float = 0.6, sync_steps: int = 512):
        self.trainer = trainer
        self.alpha = alpha
        self.sync_steps = max(1, int(sync_steps))

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.sync_steps != 0:
            return
        teacher = self.trainer.teacher_model
        if teacher is None:
            return
        student = self.trainer.accelerator.unwrap_model(self.trainer.model)
        a = self.alpha
        s_params = dict(student.named_parameters())
        for name, t_param in teacher.named_parameters():
            s_param = s_params.get(name)
            if s_param is None:
                continue
            t_param.data.mul_(1.0 - a).add_(s_param.data.to(t_param.device), alpha=a)
