"""GPU validation: vLLM generation backend produces aligned completions and the
weight-sync round-trips.

Very heavy (loads a model into the vLLM engine), so gated behind SDFT_RUN_VLLM=1
in addition to CUDA + vllm. Run on the A100:
    SDFT_RUN_VLLM=1 pytest tests/test_gpu_vllm.py -v -s
"""

import torch

from conftest import requires_cuda, requires_vllm, run_vllm

_SMALL = "Qwen/Qwen2.5-0.5B-Instruct"


@requires_cuda
@requires_vllm
@run_vllm
def test_vllm_generate_shapes_and_sync():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sdft.generation import VLLMGenerator

    tok = AutoTokenizer.from_pretrained(_SMALL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    gen = VLLMGenerator(_SMALL, gpu_memory_utilization=0.3, enable_sleep_mode=True)

    # two left-padded prompts
    tok.padding_side = "left"
    enc = tok(["What is 2+2?", "Name a color."], return_tensors="pt", padding=True)
    prompt_ids = enc["input_ids"].cuda()
    prompt_mask = enc["attention_mask"].cuda()

    comp_ids, comp_mask = gen.generate(
        prompt_ids, prompt_mask, tok, max_new_tokens=16, temperature=1.0
    )
    assert comp_ids.shape == comp_mask.shape
    assert comp_ids.shape[0] == 2
    assert int(comp_mask.sum()) > 0  # produced some real tokens

    # weight sync round-trip: load current HF weights into the engine, regenerate
    model = AutoModelForCausalLM.from_pretrained(_SMALL, torch_dtype=torch.bfloat16).cuda()
    gen.sync_weights(model)
    comp_ids2, comp_mask2 = gen.generate(
        prompt_ids, prompt_mask, tok, max_new_tokens=16, temperature=0.0
    )
    assert comp_ids2.shape[0] == 2 and int(comp_mask2.sum()) > 0
