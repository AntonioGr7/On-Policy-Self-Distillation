"""On-policy completion sampling for SDFT.

The student samples completions from its *own* policy given the prompt. Default
backend is ``transformers.generate`` (works everywhere, including the 4GB dev
box). A vLLM backend is sketched for throughput on the A100 but is optional and
not required for correctness.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def generate_on_policy(
    model,
    tokenizer,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None,
    cache_implementation: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample completions from the student.

    Args:
        model: the (unwrapped) student model in eval-safe state.
        prompt_input_ids: ``(B, P)`` left-padded prompt ids.
        prompt_attention_mask: ``(B, P)`` mask for the prompts.

    Returns:
        ``(sequences, completion_mask)`` where ``sequences`` is ``(B, P + C)``
        (prompt followed by completion, right-padded) and ``completion_mask`` is
        ``(B, P + C)`` marking the generated completion tokens (1) vs prompt and
        padding (0). The mask is shifted/consumed by the trainer to align logits.
    """
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
    )
    if top_k is not None:
        gen_kwargs["top_k"] = top_k
    if cache_implementation is not None:
        # "static" enables a compile-friendly KV cache (pair with torch.compile).
        gen_kwargs["cache_implementation"] = cache_implementation

    was_training = model.training
    model.eval()
    sequences = model.generate(
        input_ids=prompt_input_ids,
        attention_mask=prompt_attention_mask,
        **gen_kwargs,
    )
    if was_training:
        model.train()

    prompt_len = prompt_input_ids.shape[1]
    batch_size, total_len = sequences.shape

    # Completion mask: 1 for generated tokens that are not pad.
    completion_mask = torch.zeros((batch_size, total_len), dtype=torch.long, device=sequences.device)
    completion_mask[:, prompt_len:] = 1
    completion_mask = completion_mask & (sequences != tokenizer.pad_token_id).long()
    # Keep the first EOS as a valid target, but drop padding after it (handled by
    # the pad check above; EOS itself stays masked-in as a learning signal).

    return sequences, completion_mask


class VLLMGenerator:
    """Colocated vLLM backend for fast on-policy generation (A100, CUDA-only).

    vLLM runs in the same process as training (``vllm_mode=colocate``), sharing
    the GPU. Between generations the student's updated weights are pushed into
    the vLLM engine via ``load_weights``; with sleep mode enabled, vLLM releases
    its GPU memory while the training step runs.

    NOTE: vLLM's in-the-loop weight-sync API is version-sensitive. This class
    targets the stable ``LLM`` + ``llm_engine`` surface; validate against your
    installed vLLM on the A100. The transformers backend remains the default and
    is fully tested.
    """

    def __init__(
        self,
        model_name,
        gpu_memory_utilization=0.3,
        enable_sleep_mode=True,
        attention_backend="TRITON_ATTN",
    ):
        import os

        from vllm import LLM, SamplingParams  # noqa: F401  (import-time check)

        # vllm defaults to fork() for its EngineCore subprocess, which fails when
        # CUDA is already initialised in the parent (e.g. a training step ran
        # first). Force spawn so the child starts with a clean CUDA state.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # flashinfer-cubin from PyPI is compiled for CUDA 13; on CUDA 12.x hosts
        # the driver rejects it with cudaErrorInsufficientDriver. Fall back to
        # the PyTorch-native sampler which works on any supported CUDA version.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        # Run the v1 EngineCore in-process (no MP subprocess). Colocate weight
        # sync needs direct access to the model living on this process's GPU; the
        # default multiprocessing engine hides it behind an IPC client.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        # The bundled vllm-flash-attn binary is built against a newer CUDA
        # runtime than CUDA 12.x drivers support, so the auto-selected FLASH_ATTN
        # backend dies with "CUDA driver version is insufficient for CUDA runtime
        # version" during cudagraph capture. TRITON_ATTN is JIT-compiled locally
        # and works on any supported CUDA. Pass attention_backend=None to restore
        # vLLM's automatic selection.
        self.model_name = model_name
        llm_kwargs = dict(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_sleep_mode=enable_sleep_mode,
            dtype="bfloat16",
        )
        if attention_backend is not None:
            llm_kwargs["attention_backend"] = attention_backend
        self.llm = LLM(**llm_kwargs)
        self._SamplingParams = SamplingParams

    def sync_weights(self, model) -> None:
        """Load the current training weights into the vLLM engine."""
        state = model.state_dict()

        def _load(vllm_model):
            vllm_model.load_weights((name, p) for name, p in state.items())

        # apply_model runs the closure on the model inside the worker; with the
        # in-process engine the state_dict tensors are already on this GPU.
        self.llm.apply_model(_load)

    @torch.no_grad()
    def generate(
        self, prompt_ids, prompt_attention_mask, tokenizer, *,
        max_new_tokens, temperature=1.0, top_p=1.0, top_k=None,
    ):
        """Generate completions, returning ``(completion_ids, completion_mask)``
        right-padded to a common length — matching the transformers backend."""
        device = prompt_ids.device
        # detokenize the (left-padded) prompts back to text for vLLM
        prompts = []
        for ids, m in zip(prompt_ids, prompt_attention_mask):
            real = ids[m.bool()]
            prompts.append(tokenizer.decode(real, skip_special_tokens=False))

        sp = self._SamplingParams(
            n=1, temperature=temperature, top_p=top_p,
            top_k=top_k if top_k is not None else -1, max_tokens=max_new_tokens,
        )
        outputs = self.llm.generate(prompts, sp)
        comp_token_lists = [list(o.outputs[0].token_ids) for o in outputs]

        max_c = max((len(c) for c in comp_token_lists), default=1)
        pad_id = tokenizer.pad_token_id
        comp_ids = torch.full((len(comp_token_lists), max_c), pad_id, dtype=torch.long, device=device)
        comp_mask = torch.zeros((len(comp_token_lists), max_c), dtype=torch.long, device=device)
        for i, toks in enumerate(comp_token_lists):
            if toks:
                comp_ids[i, : len(toks)] = torch.tensor(toks, device=device)
                comp_mask[i, : len(toks)] = 1
        return comp_ids, comp_mask
