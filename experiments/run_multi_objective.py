import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run Multi-Objective DPO sweeps")
    # Common training args
    p.add_argument("--model_name", type=str, default="gpt2")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--lr", type=float, default=1e-5)
    # Efficiency / stability
    p.add_argument("--dtype", type=str, choices=["auto", "fp16", "bf16", "fp32"], default=None)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--eval_interval", type=int, default=0)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # Multi-objective sweeps
    p.add_argument(
        "--weights_list",
        type=str,
        default="base=0.7,brevity=0.3;base=0.9,brevity=0.1;base=1.0",
        help="Semicolon-separated mo_weights specs. Example: 'base=0.7,brevity=0.3;base=1.0'",
    )
    p.add_argument(
        "--brevity_coefs",
        type=str,
        default="0.0,0.01",
        help="Comma-separated brevity coefficients. Example: '0.0,0.01,0.02'",
    )
    p.add_argument("--base_output", type=str, default="outputs/mo_sweep")
    return p.parse_args()


def main():
    args = parse_args()
    weights_specs = [w.strip() for w in args.weights_list.split(";") if w.strip()]
    brevity_coefs = [b.strip() for b in args.brevity_coefs.split(",") if b.strip()]

    Path(args.base_output).mkdir(parents=True, exist_ok=True)

    for ws in weights_specs:
        for bc in brevity_coefs:
            tag = ws.replace("=", "").replace(",", "_").replace(" ", "")
            out_dir = f"{args.base_output}/{args.model_name.replace('/','_')}_{tag}_brc{bc}"

            cmd = [
                sys.executable, "DPO/train_dpo.py",
                "--model_name", args.model_name,
                "--batch_size", str(args.batch_size),
                "--num_epochs", str(args.num_epochs),
                "--max_length", str(args.max_length),
                "--device", args.device,
                "--lr", str(args.lr),
                # MO switches
                "--multi_objective",
                "--mo_weights", ws,
                "--brevity_coef", bc,
                # Efficiency
                "--grad_accum", str(args.grad_accum),
                "--eval_interval", str(args.eval_interval),
                "--output_dir", out_dir,
            ]
            if args.dtype is not None:
                cmd += ["--dtype", args.dtype]
            if args.grad_ckpt:
                cmd += ["--grad_ckpt"]
            if args.use_lora:
                cmd += ["--use_lora", "--lora_r", str(args.lora_r), "--lora_alpha", str(args.lora_alpha), "--lora_dropout", str(args.lora_dropout)]

            print(f"\n=== Running: {out_dir} ===")
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

