# On-Policy Self-Distillation (SDFT)

A standardized, cross-model framework for **On-Policy Self-Distillation
Fine-Tuning** — transferring task knowledge into a language model while
preserving the knowledge it already has (continual learning).

Based on the SDFT method (project page: <https://self-distillation.github.io/SDFT>,
reference implementation: <https://github.com/idanshen/Self-Distillation>),
reimplemented as a clean, config-driven framework so the same recipe runs across
model families and scales, on a **single A100**.

## The idea in one paragraph

The model is its **own teacher**. The *teacher* is shown the gold demonstration
**in-context**; the *student* sees only the prompt. We sample a completion
**on-policy from the student**, then minimize a per-token **forward-KL** (or
generalized JSD) between the student's and the demonstration-conditioned
teacher's distributions over *the student's own tokens*. Because the student
learns to recover from its own trajectories rather than memorizing fixed expert
tokens, it acquires the new task while **forgetting far less** than ordinary SFT.

```
prompt x ──► student  ──sample──►  ŷ (on-policy completion)
                                     │
prompt x + demo y* (in context) ──► teacher ──► p_teacher(·|x,y*)
                                     │
            loss = JSD_alpha( student(·|x) ‖ teacher(·|x,y*) )  over ŷ
```

`alpha=0` → forward KL (paper default), `alpha=1` → reverse KL, in between → JSD.

## Install

```bash
uv venv && source .venv/bin/activate
uv pip install -e .            # core (CPU/dev box)
uv pip install -e ".[train]"   # + deepspeed, bitsandbytes, wandb (A100)
uv pip install -e ".[eval]"    # + lm-eval-harness (forgetting probe)
uv pip install -e ".[vllm]"    # + vLLM (optional, faster generation)
```

## Quickstart

```bash
# Sanity check on a tiny model (CPU or the 4GB dev box):
python scripts/train.py --config configs/smoke_qwen0.5b.yaml

# Real single-task run on an A100:
python scripts/train.py --config configs/qwen2.5-3b_a100-40g.yaml

# Continual-learning study (the headline deliverable):
python scripts/train_continual.py --sequence experiments/continual_example.yaml
```

## Switching model family

One line. Set `model.family` (and optionally `model.name`); per-family defaults
— including how the demonstration is injected for that chat template — come from
`configs/families/<family>.yaml`. Supported out of the box: `qwen`, `llama`,
`gemma`, `mistral`. Add a new family by extending `FAMILY_REGISTRY` in
[sdft/models.py](sdft/models.py).

## Hardware matrix (full fine-tuning, student + frozen teacher)

| GPU | Recommended | Notes |
|---|---|---|
| 4GB dev box | Qwen2.5-0.5B smoke | `configs/smoke_qwen0.5b.yaml`, transformers generation, no bf16 |
| A100 40GB | up to ~3B | `adamw_bnb_8bit` + grad checkpointing; 7B needs ZeRO-3 + CPU offload |
| A100 80GB | 7B | ZeRO-2 + 8-bit Adam + grad checkpointing; 14B → ZeRO-3 + offload |

Full-FT keeps two model copies (student + teacher) in memory, so 8-bit Adam,
gradient checkpointing, and (for 7B on 40GB) DeepSpeed ZeRO + CPU offload are the
levers that make a single A100 work. See `configs/ds_zero*.json`.

### Memory / speed optimizations

On-policy distillation has two memory bottlenecks beyond ordinary fine-tuning:
the **second (teacher) model copy** and the **doubled full-vocab logits** in the
loss. The framework targets both — all opt-in via config:

| Lever | Config | What it buys |
|---|---|---|
| **Quantized teacher** (nf4/int8) | `model.teacher_quantization: nf4` | 7B teacher ~14GB → ~4GB. *Disables EMA sync* (frozen weights). |
| **Liger fused-linear JSD** | `sdft.use_liger_jsd: true` | Never materializes 152K-vocab logits; big memory + speed win. Dense fallback if Liger absent. |
| **Top-k KL** | `sdft.loss_top_k: 256` | KL over the teacher's top-k tokens only; denoises + trims the loss tensor. |
| **8-bit / paged Adam** | `sdft.optim: paged_adamw_8bit` | Halves optimizer state; paged variant absorbs OOM spikes. |
| **vLLM generation** | `sdft.generation_backend: vllm` | Fast on-policy sampling (colocate + sleep mode + weight sync). |
| **Compile + static cache** | `sdft.compile_generation: true`, `cache_implementation: static` | Faster generation loop. |

These stack: `configs/qwen2.5-7b_a100-40g_optimized.yaml` fits **7B full-FT on a
40GB A100** with nf4 teacher + Liger + paged 8-bit Adam.

> The vLLM in-loop weight-sync and the quantized paths are CUDA-only and
> version-sensitive — validate them on the A100. The transformers backend and
> the dense loss are the fully-tested defaults.

> Note: with DeepSpeed **ZeRO-3** the separate teacher is not sharded; prefer
> ZeRO-2 + offload when the teacher fits, or enable EMA sync so the teacher can
> later be replaced by the (sharded) student snapshot. This is a known trade-off
> inherited from full-FT self-distillation.

## Repository layout

```
sdft/        config, model registry, data, generation, losses, trainer, callbacks, cli
configs/     run configs + families/ + DeepSpeed json
scripts/     train, train_continual, eval_forgetting, eval_task
experiments/ continual-learning protocol + example sequence
tests/       loss correctness, data schema, gated smoke train
```

## Verify

```bash
pytest                       # loss + data/config tests (CPU, no download)
SDFT_RUN_SMOKE=1 pytest tests/test_smoke_train.py -s   # end-to-end (downloads 0.5B)
```

GPU-path tests (skip on CPU; run on the A100 after `uv pip install -e ".[train,vllm]"`):

```bash
pytest tests/test_gpu_liger.py            # Liger fused JSD == dense loss
pytest tests/test_gpu_quantized_teacher.py  # nf4 teacher loads + disables EMA
SDFT_RUN_VLLM=1 pytest tests/test_gpu_vllm.py -s   # vLLM generate + weight sync
```

## Goals this repo is built to study

1. **Standardize** the transfer recipe across model families (one config switch).
2. **Measure** how much task knowledge transfers per model class (`eval_task.py`).
3. **Demonstrate continual learning** without destroying internal knowledge —
   SDFT vs. an SFT baseline on a fixed forgetting suite (`eval_forgetting.py`,
   `train_continual.py`). See [experiments/README.md](experiments/README.md).
