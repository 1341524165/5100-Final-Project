# models.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Tuple
import torch

# 如果你要用 LoRA，可以解开下面注释并在 config 里 use_lora=True
# from peft import LoraConfig, get_peft_model


def load_tokenizer_and_models(
    model_name: str,
    ref_model_name: str | None = None,
    device: str = "cuda",
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> Tuple[PreTrainedTokenizer, PreTrainedModel, PreTrainedModel]:
    """
    返回 tokenizer, policy_model, reference_model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None
    )

    if ref_model_name is None:
        ref_model_name = model_name

    reference = AutoModelForCausalLM.from_pretrained(
        ref_model_name,
        device_map="auto" if device == "cuda" else None
    )
    reference.requires_grad_(False)
    reference.eval()

    if use_lora:
        # 你可以根据自己的需求修改 LoRA target modules
        # lora_config = LoraConfig(
        #     r=lora_r,
        #     lora_alpha=lora_alpha,
        #     lora_dropout=lora_dropout,
        #     bias="none",
        #     task_type="CAUSAL_LM",
        # )
        # policy = get_peft_model(policy, lora_config)
        # print(policy.print_trainable_parameters())
        raise NotImplementedError("LoRA 支持在此处自行打开/修改")

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