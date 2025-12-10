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
    noise_rate: float = 0.0 # 标签噪声比例 (0.0 - 1.0)
    eval_interval: int = 500
    logging_steps: int = 50  # 训练中每隔多少步写一次 step 级指标
    # Multi-objective DPO
    multi_objective: bool = False
    mo_weights: dict[str, float] | None = None  # 例如 {"base": 0.7, "brevity": 0.3}
    brevity_coef: float = 0.0  # 用于 brevity 目标的长度惩罚系数
    # Constraints
    kl_weight: float = 0.0  # KL（log-ratio L2 近似）约束权重，0 表示关闭

    # 设备
    device: str = "cuda"

    # 日志 & 保存
    output_dir: str = "outputs/dpo_run"
    save_every_epoch: bool = True
    seed: int = 42