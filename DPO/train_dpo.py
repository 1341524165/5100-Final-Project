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
    )

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

    train_dataset = PreferenceDataset(train_pairs, tokenizer, config.max_length)
    val_dataset = PreferenceDataset(val_pairs, tokenizer, config.max_length) if val_pairs is not None else None

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False) if val_dataset is not None else None

    print(f"Loaded HH-RLHF dataset | train: {len(train_dataset)} samples"
          + (f", val: {len(val_dataset)} samples" if val_dataset is not None else ", val: None"))

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