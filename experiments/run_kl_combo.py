import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run single-objective DPO+KL and multi-objective DPO+KL")
    # Common training args
    p.add_argument("--model_name", type=str, default="gpt2")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--lr", type=float, default=5e-5)
    # Efficiency / stability
    p.add_argument("--dtype", type=str, choices=["auto", "fp16", "bf16", "fp32"], default=None)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--eval_interval", type=int, default=0)
    # LoRA
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # KL constraint
    p.add_argument("--kl_weight", type=float, default=0.05)
    # Multi-objective settings
    p.add_argument("--mo_weights", type=str, default="base=0.8,brevity=0.2",
                   help="Single spec for MO run, e.g. 'base=0.8,brevity=0.2'")
    p.add_argument("--weights_list", type=str, default="",
                   help="Semicolon-separated multiple specs, e.g. 'base=0.8,brevity=0.2;base=0.7,brevity=0.3'")
    p.add_argument("--brevity_coef", type=float, default=0.01)
    # Output
    p.add_argument("--base_output", type=str, default="outputs/kl_combo")
    return p.parse_args()


def build_common_cmd(args, out_dir: str):
    cmd = [
        sys.executable, "DPO/train_dpo.py",
        "--model_name", args.model_name,
        "--batch_size", str(args.batch_size),
        "--num_epochs", str(args.num_epochs),
        "--max_length", str(args.max_length),
        "--device", args.device,
        "--lr", str(args.lr),
        "--grad_accum", str(args.grad_accum),
        "--eval_interval", str(args.eval_interval),
        "--kl_weight", str(args.kl_weight),
        "--output_dir", out_dir,
    ]
    if args.dtype is not None:
        cmd += ["--dtype", args.dtype]
    if args.grad_ckpt:
        cmd += ["--grad_ckpt"]
    if args.use_lora:
        cmd += ["--use_lora", "--lora_r", str(args.lora_r), "--lora_alpha", str(args.lora_alpha), "--lora_dropout", str(args.lora_dropout)]
    return cmd


def main():
    args = parse_args()
    Path(args.base_output).mkdir(parents=True, exist_ok=True)

    # 1) Single-objective DPO + KL
    out_base = f"{args.base_output}/{args.model_name.replace('/','_')}_baseKL"
    cmd_base = build_common_cmd(args, out_base)
    print(f"\n=== Running single-objective DPO+KL ===\n{' '.join(cmd_base)}")
    subprocess.run(cmd_base, check=True)

    # 2) Multi-objective DPO + KL
    if args.weights_list:
        specs = [w.strip() for w in args.weights_list.split(";") if w.strip()]
    else:
        specs = [args.mo_weights]

    for ws in specs:
        tag = ws.replace("=", "").replace(",", "_").replace(" ", "")
        out_mo = f"{args.base_output}/{args.model_name.replace('/','_')}_MO_{tag}_brc{args.brevity_coef}"
        cmd_mo = build_common_cmd(args, out_mo)
        cmd_mo += ["--multi_objective", "--mo_weights", ws, "--brevity_coef", str(args.brevity_coef)]
        print(f"\n=== Running multi-objective DPO+KL ===\n{' '.join(cmd_mo)}")
        subprocess.run(cmd_mo, check=True)


if __name__ == "__main__":
    main()

