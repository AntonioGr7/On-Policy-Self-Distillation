"""On-Policy Self-Distillation Fine-Tuning (SDFT).

A standardized, cross-model framework for transferring task knowledge into a
language model via on-policy self-distillation, while preserving the model's
pre-existing internal knowledge (continual learning).

The model is its own teacher: the teacher is conditioned on the gold
demonstration in-context, the student sees only the prompt, completions are
sampled on-policy from the student, and a forward-KL / JSD loss distills the
teacher's distribution into the student.
"""

from sdft.config import DataConfig, ModelConfig, RunConfig, SDFTConfig, load_config

__all__ = [
    "DataConfig",
    "ModelConfig",
    "RunConfig",
    "SDFTConfig",
    "load_config",
]

__version__ = "0.1.0"
