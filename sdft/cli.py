"""Build an SDFTTrainer from a RunConfig, and the ``sdft-train`` entrypoint."""

from __future__ import annotations

import argparse
import math

import torch
from transformers import TrainingArguments

from sdft.callbacks import EMATeacherSyncCallback
from sdft.config import RunConfig, load_config
from sdft.data import load_sdft_dataset
from sdft.models import load_student_and_teacher
from sdft.trainer import SDFTCollator, SDFTTrainer


def _training_arguments(run: RunConfig, dataset_size: int = 0) -> TrainingArguments:
    s = run.sdft
    cuda = torch.cuda.is_available()
    # warmup_ratio is deprecated in transformers v5.2; compute warmup_steps instead.
    if s.max_steps > 0:
        total_steps = s.max_steps
    else:
        steps_per_epoch = math.ceil(
            dataset_size / (s.per_device_train_batch_size * s.gradient_accumulation_steps)
        )
        total_steps = int(steps_per_epoch * s.num_train_epochs)
    warmup_steps = math.ceil(s.warmup_ratio * total_steps)
    return TrainingArguments(
        output_dir=s.output_dir,
        run_name=run.run_name,
        learning_rate=s.learning_rate,
        num_train_epochs=s.num_train_epochs,
        max_steps=s.max_steps,
        per_device_train_batch_size=s.per_device_train_batch_size,
        gradient_accumulation_steps=s.gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        lr_scheduler_type=s.lr_scheduler_type,
        weight_decay=s.weight_decay,
        max_grad_norm=s.max_grad_norm,
        bf16=bool(s.bf16 and cuda),  # bf16 only makes sense on the A100
        gradient_checkpointing=s.gradient_checkpointing and cuda,
        optim=s.optim,
        logging_steps=s.logging_steps,
        save_steps=s.save_steps,
        save_total_limit=s.save_total_limit,
        seed=s.seed,
        deepspeed=s.deepspeed,
        report_to=[s.report_to] if s.report_to != "none" else [],
        remove_unused_columns=False,  # our collator consumes the raw columns
    )


def build_trainer(run: RunConfig) -> SDFTTrainer:
    """Construct a ready-to-train SDFTTrainer from a RunConfig."""
    student, teacher, tokenizer = load_student_and_teacher(run.model)
    dataset = load_sdft_dataset(run.data)
    collator = SDFTCollator(
        tokenizer=tokenizer,
        model_cfg=run.model,
        max_prompt_length=run.data.max_prompt_length,
        max_completion_length=run.data.max_completion_length,
    )
    args = _training_arguments(run, len(dataset))

    # A quantized teacher has frozen, non-updatable weights -> EMA sync is
    # impossible. Disable it (with a warning) rather than fail mid-run.
    sync_ref_model = run.sdft.sync_ref_model
    if sync_ref_model and run.model.teacher_quantization:
        import warnings

        warnings.warn(
            "sync_ref_model=True is incompatible with a quantized teacher "
            f"(teacher_quantization='{run.model.teacher_quantization}'); disabling EMA sync."
        )
        sync_ref_model = False

    if run.sdft.compile_generation:
        student = torch.compile(student)

    trainer = SDFTTrainer(
        model=student,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
        teacher_model=teacher,
        sdft_cfg=run.sdft,
        data_cfg=run.data,
        model_name=run.model.name,
    )
    if sync_ref_model and teacher is not None:
        trainer.add_callback(
            EMATeacherSyncCallback(
                trainer,
                alpha=run.sdft.ref_model_mixup_alpha,
                sync_steps=run.sdft.ref_model_sync_steps,
            )
        )
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model with On-Policy SDFT.")
    parser.add_argument("--config", required=True, help="Path to a run YAML config.")
    parser.add_argument("--output_dir", default=None, help="Override sdft.output_dir.")
    parser.add_argument("--model_name", default=None, help="Override model.name.")
    parser.add_argument("--max_steps", type=int, default=None, help="Override sdft.max_steps.")
    args = parser.parse_args()

    run = load_config(args.config)
    if args.output_dir:
        run.sdft.output_dir = args.output_dir
    if args.model_name:
        run.model.name = args.model_name
    if args.max_steps is not None:
        run.sdft.max_steps = args.max_steps

    trainer = build_trainer(run)
    trainer.train()
    trainer.save_model(run.sdft.output_dir)
    trainer.processing_class.save_pretrained(run.sdft.output_dir)
    print(f"[sdft] training complete -> {run.sdft.output_dir}")


if __name__ == "__main__":
    main()
