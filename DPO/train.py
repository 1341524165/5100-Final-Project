import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm  # type: ignore

from config import DPOConfig
from losses import dpo_loss
from utils import save_policy_model


def train_one_epoch(
    config: DPOConfig,
    policy,
    reference,
    tokenizer,
    train_loader,
    optimizer,
    scheduler,
    epoch: int,
    device: str = "cuda",
    val_loader=None,
    eval_interval: int = 0,
):
    policy.train()
    total_loss = 0.0
    global_step = 0

    num_batches = len(train_loader)
    progress = tqdm(total=num_batches, desc=f"Epoch {epoch}")

    for step, batch in enumerate(train_loader):
        # 把 batch 移到 device
        for k in ["chosen_input", "rejected_input"]:
            for kk in batch[k]:
                batch[k][kk] = batch[k][kk].to(device)

        loss = dpo_loss(policy, reference, batch, beta=config.beta)
        loss = loss / config.gradient_accumulation_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)

        if (step + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * config.gradient_accumulation_steps
        global_step += 1

        if global_step % 50 == 0:
            print(f"[Epoch {epoch}] Step {global_step} | Loss {total_loss / global_step:.4f}")

        # 更新进度条
        current_lr = optimizer.param_groups[0]["lr"]
        postfix = {"loss": f"{(total_loss / global_step):.4f}"}
        postfix["lr"] = f"{current_lr:.2e}"
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(postfix)
        if hasattr(progress, "update"):
            progress.update(1)

        # 按间隔进行验证评估（如果提供了验证集）
        if eval_interval > 0 and val_loader is not None and (global_step % eval_interval == 0):
            val_loss = evaluate(config, policy, reference, val_loader, device=device)
            if val_loss is not None:
                print(f"[Epoch {epoch}] Step {global_step} | Val loss {val_loss:.4f}")

    if hasattr(progress, "close"):
        progress.close()

    avg_loss = total_loss / global_step
    print(f"Epoch {epoch} finished. Avg loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(
    config: DPOConfig,
    policy,
    reference,
    val_loader,
    device: str = "cuda",
):
    if val_loader is None:
        print("No validation set provided, skip evaluation.")
        return None

    policy.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in val_loader:
            for k in ["chosen_input", "rejected_input"]:
                for kk in batch[k]:
                    batch[k][kk] = batch[k][kk].to(device)

            loss = dpo_loss(policy, reference, batch, beta=config.beta)
            total_loss += loss.item()
            steps += 1

    avg_loss = total_loss / steps if steps > 0 else None
    print(f"Validation loss: {avg_loss:.4f}" if avg_loss is not None else "No validation steps.")
    return avg_loss


def train_dpo(
    config: DPOConfig,
    policy,
    reference,
    tokenizer,
    train_loader,
    val_loader=None,
):
    device = config.device
    policy.to(device)
    reference.to(device)

    num_update_steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    t_total = num_update_steps_per_epoch * config.num_epochs

    optimizer = AdamW(policy.parameters(),
                      lr=config.learning_rate,
                      weight_decay=config.weight_decay)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=t_total
    )

    best_val_loss = None

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_one_epoch(
            config, policy, reference, tokenizer,
            train_loader, optimizer, scheduler, epoch, device=device,
            val_loader=val_loader, eval_interval=config.eval_interval
        )

        val_loss = evaluate(config, policy, reference, val_loader, device=device)

        # 简单的 best 模型保存逻辑
        if val_loss is not None:
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                save_policy_model(policy, tokenizer, config.output_dir + "/best")

        if config.save_every_epoch:
            save_policy_model(policy, tokenizer, f"{config.output_dir}/epoch_{epoch}")