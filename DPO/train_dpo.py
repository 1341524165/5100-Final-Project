import argparse
import os

from config import DPOConfig
from models import load_tokenizer_and_models
from data import PreferenceDataset
from train import train_dpo
from utils import set_seed, ensure_dir
from datasets import load_dataset  # type: ignore
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train DPO model")
    parser.add_argument("--output_dir", type=str, default="outputs/dpo_run")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    # OOM 缓解相关
    parser.add_argument("--dtype", type=str, choices=["auto", "fp16", "bf16", "fp32"], default=None)
    parser.add_argument("--grad_accum", type=int, default=1, help="gradient_accumulation_steps")
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--grad_ckpt", action="store_true", help="enable gradient checkpointing")
    # LoRA 配置
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # Noise robustness
    parser.add_argument("--noise_rate", type=float, default=0.0, help="label noise rate (0.0-1.0)")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # 初始化配置
    config = DPOConfig(
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        device=args.device,
    )
    # 透传更多配置
    config.gradient_accumulation_steps = args.grad_accum
    config.eval_interval = args.eval_interval
    config.use_lora = args.use_lora
    config.lora_r = args.lora_r
    config.lora_alpha = args.lora_alpha
    config.lora_dropout = args.lora_dropout
    config.noise_rate = args.noise_rate

    set_seed(config.seed)

    # 将 batch_size 附加到输出目录名中，便于区分不同实验
    config.output_dir = f"{config.output_dir}_bs{config.batch_size}"
    ensure_dir(config.output_dir)
    print(f"Output directory: {config.output_dir}")

    # 加载 tokenizer & 模型
    tokenizer, policy, reference = load_tokenizer_and_models(
        model_name=config.model_name,
        ref_model_name=config.ref_model_name,
        device=config.device,
        use_lora=config.use_lora,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        torch_dtype=args.dtype,
    )
    if args.grad_ckpt:
        try:
            policy.gradient_checkpointing_enable()
            policy.config.use_cache = False
            print("Enabled gradient checkpointing for policy model.")
        except Exception as e:
            print(f"Enable gradient checkpointing failed: {e}")

    # 构建 DataLoader：直接使用 HuggingFace 的 Anthropic/hh-rlhf
    ds = load_dataset("Anthropic/hh-rlhf")

    def to_pairs(split):
        return [{
            "prompt": "",
            "chosen": ex["chosen"],
            "rejected": ex["rejected"],
        } for ex in split]

    train_pairs = to_pairs(ds["train"])
    val_pairs = to_pairs(ds["test"]) if "test" in ds else None

    # Create datasets with noise injection (only for training)
    train_dataset = PreferenceDataset(train_pairs, tokenizer, config.max_length, noise_rate=config.noise_rate)
    val_dataset = PreferenceDataset(val_pairs, tokenizer, config.max_length, noise_rate=0.0) if val_pairs is not None else None

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False) if val_dataset is not None else None

    print(f"Loaded HH-RLHF dataset | train: {len(train_dataset)} samples"
          + (f", val: {len(val_dataset)} samples" if val_dataset is not None else ", val: None"))
    print(f"Noise rate: {config.noise_rate:.1%} (applied to training set only)")

    # 开始训练
    train_dpo(
        config=config,
        policy=policy,
        reference=reference,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
    )


if __name__ == "__main__":
    main()