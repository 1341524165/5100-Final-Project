from dataclasses import dataclass

@dataclass
class DPOConfig:
    # 模型
    model_name: str = "meta-llama/Llama-2-7b-hf"
    ref_model_name: str | None = None  # 默认与 model_name 相同
    use_lora: bool = False             # 你之后想接 LoRA 可以用
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # 训练
    # train_path: str = "data/train.jsonl"
    # val_path: str | None = None
    max_length: int = 512
    batch_size: int = 1
    num_epochs: int = 3
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # DPO
    beta: float = 0.1   # 温度/强度
    eval_interval: int = 500

    # 设备
    device: str = "cuda"

    # 日志 & 保存
    output_dir: str = "outputs/dpo_run"
    save_every_epoch: bool = True
    seed: int = 42