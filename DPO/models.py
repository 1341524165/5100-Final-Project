# models.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Tuple, Optional
import torch

from peft import LoraConfig, get_peft_model


def load_tokenizer_and_models(
    model_name: str,
    ref_model_name: str | None = None,
    device: str = "cuda",
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    torch_dtype: Optional[str] = None,
) -> Tuple[PreTrainedTokenizer, PreTrainedModel, PreTrainedModel]:
    """
    返回 tokenizer, policy_model, reference_model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 解析 dtype
    dtype_map: dict[str, object] = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "auto": "auto",
    }
    hf_dtype = None
    if torch_dtype is not None:
        hf_dtype = dtype_map.get(torch_dtype, None)

    policy = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=hf_dtype if hf_dtype is not None else None,
    )

    if ref_model_name is None:
        ref_model_name = model_name

    reference = AutoModelForCausalLM.from_pretrained(
        ref_model_name,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=hf_dtype if hf_dtype is not None else None,
    )
    reference.requires_grad_(False)
    reference.eval()

    if use_lora:
        # 选择一组通用的 target modules，覆盖 LLaMA / GPT2 常见命名
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "c_attn", "c_proj", "c_fc",
        ]
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        policy = get_peft_model(policy, lora_config)
        try:
            policy.print_trainable_parameters()
        except Exception:
            pass

    policy.to(device)
    reference.to(device)

    return tokenizer, policy, reference


@torch.no_grad()
def generate_answer(model: PreTrainedModel,
                    tokenizer: PreTrainedTokenizer,
                    prompt: str,
                    device: str = "cuda",
                    max_new_tokens: int = 200) -> str:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)