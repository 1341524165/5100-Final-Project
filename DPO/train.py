import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm  # type: ignore
import os
import csv

from config import DPOConfig
from losses import dpo_loss, multi_objective_dpo_loss, kl_penalty_logratio
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

        if config.multi_objective and config.mo_weights:
            loss, _ = multi_objective_dpo_loss(
                policy, reference, batch,
                beta=config.beta,
                weights=config.mo_weights,
                brevity_coef=config.brevity_coef,
                kl_weight=config.kl_weight,
            )
        else:
            loss = dpo_loss(policy, reference, batch, beta=config.beta)
            if config.kl_weight > 0.0:
                loss = loss + config.kl_weight * kl_penalty_logratio(policy, reference, batch)
        loss = loss / config.gradient_accumulation_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)

        if (step + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * config.gradient_accumulation_steps
        global_step += 1

        if global_step % config.logging_steps == 0:
            avg_tr_loss = total_loss / global_step
            print(f"[Epoch {epoch}] Step {global_step} | Loss {avg_tr_loss:.4f}")
            # 记录 step 级指标（训练过程中）
            try:
                metrics_steps_path = os.path.join(config.output_dir, "metrics_steps.csv")
                file_exists = os.path.exists(metrics_steps_path)
                with open(metrics_steps_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["epoch", "global_step", "step_in_epoch", "train_loss", "lr", "val_loss", "val_accuracy", "val_reward_margin"])
                    current_lr = optimizer.param_groups[0]["lr"]
                    writer.writerow([
                        epoch,
                        global_step,
                        step + 1,
                        f"{avg_tr_loss:.6f}",
                        f"{current_lr:.6e}",
                        "", "", ""  # 占位，若无即时验证
                    ])
            except Exception as e:
                print(f"Warning: failed to write metrics_steps.csv: {e}")

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
            val_stats = evaluate(config, policy, reference, val_loader, device=device)
            if val_stats is not None and val_stats.get("loss") is not None:
                print(f"[Epoch {epoch}] Step {global_step} | Val loss {val_stats['loss']:.4f} | Val acc {val_stats['accuracy']:.3f} | Val margin {val_stats['reward_margin']:.4f}")
                # 评估时也将 step 级指标写入（含验证项）
                try:
                    metrics_steps_path = os.path.join(config.output_dir, "metrics_steps.csv")
                    file_exists = os.path.exists(metrics_steps_path)
                    with open(metrics_steps_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["epoch", "global_step", "step_in_epoch", "train_loss", "lr", "val_loss", "val_accuracy", "val_reward_margin"])
                        current_lr = optimizer.param_groups[0]["lr"]
                        avg_tr_loss = total_loss / global_step
                        writer.writerow([
                            epoch,
                            global_step,
                            step + 1,
                            f"{avg_tr_loss:.6f}",
                            f"{current_lr:.6e}",
                            f"{val_stats.get('loss'):.6f}" if val_stats.get("loss") is not None else "",
                            f"{val_stats.get('accuracy'):.6f}" if val_stats.get("accuracy") is not None else "",
                            f"{val_stats.get('reward_margin'):.6f}" if val_stats.get("reward_margin") is not None else "",
                        ])
                except Exception as e:
                    print(f"Warning: failed to write metrics_steps.csv: {e}")

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
    total_correct = 0
    total_examples = 0
    total_reward_margin = 0.0

    with torch.no_grad():
        for batch in val_loader:
            for k in ["chosen_input", "rejected_input"]:
                for kk in batch[k]:
                    batch[k][kk] = batch[k][kk].to(device)

            # 计算损失（与训练保持一致的模式）
            if config.multi_objective and config.mo_weights:
                mo_loss, mo_stats = multi_objective_dpo_loss(
                    policy, reference, batch,
                    beta=config.beta,
                    weights=config.mo_weights,
                    brevity_coef=config.brevity_coef,
                    kl_weight=config.kl_weight,
                )
                loss = mo_loss
            else:
                loss = dpo_loss(policy, reference, batch, beta=config.beta)
                if config.kl_weight > 0.0:
                    loss = loss + config.kl_weight * kl_penalty_logratio(policy, reference, batch)
            total_loss += loss.item()
            steps += 1

            # 统计偏好准确率与奖励边际
            # 复用 losses.dpo_loss 的内部逻辑：需要各自序列对数似然
            from losses import sequence_log_prob  # 局部导入避免循环
            c_ids = batch["chosen_input"]["input_ids"]
            c_mask = batch["chosen_input"]["attention_mask"]
            r_ids = batch["rejected_input"]["input_ids"]
            r_mask = batch["rejected_input"]["attention_mask"]

            pi_c = sequence_log_prob(policy, c_ids, c_mask)
            pi_r = sequence_log_prob(policy, r_ids, r_mask)
            ref_c = sequence_log_prob(reference, c_ids, c_mask)
            ref_r = sequence_log_prob(reference, r_ids, r_mask)

            # 偏好准确率：policy 是否更偏好 chosen
            correct = (pi_c - pi_r > 0).sum().item()
            total_correct += correct
            total_examples += pi_c.shape[0]

            # 奖励边际（DPO 中的隐式奖励差）：beta*(pi_c - ref_c) - beta*(pi_r - ref_r)
            chosen_rewards = config.beta * (pi_c - ref_c)
            rejected_rewards = config.beta * (pi_r - ref_r)
            margin = (chosen_rewards - rejected_rewards).mean().item()
            total_reward_margin += margin

    avg_loss = total_loss / steps if steps > 0 else None
    accuracy = (total_correct / total_examples) if total_examples > 0 else None
    avg_margin = (total_reward_margin / steps) if steps > 0 else None

    if avg_loss is not None:
        print(f"Validation loss: {avg_loss:.4f} | acc: {accuracy:.3f} | margin: {avg_margin:.4f}")
    else:
        print("No validation steps.")

    return {"loss": avg_loss, "accuracy": accuracy, "reward_margin": avg_margin}


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

        val_stats = evaluate(config, policy, reference, val_loader, device=device)
        val_loss = None if val_stats is None else val_stats.get("loss")

        # 简单的 best 模型保存逻辑
        if val_loss is not None:
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                save_policy_model(policy, tokenizer, config.output_dir + "/best")

        if config.save_every_epoch:
            save_policy_model(policy, tokenizer, f"{config.output_dir}/epoch_{epoch}")

        # 记录指标到 CSV，方便后期画图
        try:
            metrics_path = os.path.join(config.output_dir, "metrics.csv")
            file_exists = os.path.exists(metrics_path)
            with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_reward_margin"])
                writer.writerow([
                    epoch,
                    f"{train_loss:.6f}" if train_loss is not None else "",
                    f"{val_stats.get('loss'):.6f}" if val_stats and val_stats.get("loss") is not None else "",
                    f"{val_stats.get('accuracy'):.6f}" if val_stats and val_stats.get("accuracy") is not None else "",
                    f"{val_stats.get('reward_margin'):.6f}" if val_stats and val_stats.get("reward_margin") is not None else "",
                ])
        except Exception as e:
            print(f"Warning: failed to write metrics.csv: {e}")