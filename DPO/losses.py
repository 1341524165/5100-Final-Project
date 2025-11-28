# losses.py
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


def sequence_log_prob(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    计算 log p(y|x)：
    - input_ids: [B, L]
    - attention_mask: [B, L]
    返回: [B] 每个样本的 log prob 和
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [B, L, V]

    # 用下一个 token 作为标签
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # mask 也对齐
    shift_mask = attention_mask[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)  # [B, L-1, V]

    # 取每个位置标签对应的 log prob
    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # [B, L-1]

    # 把 pad 的位置 mask 掉
    token_log_probs = token_log_probs * shift_mask

    # 对序列求和得到 log p(y|x)
    seq_log_prob = token_log_probs.sum(dim=-1)  # [B]
    return seq_log_prob


def dpo_loss(
    policy: PreTrainedModel,
    reference: PreTrainedModel,
    batch: dict,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    DPO 损失:
    L = - E[ log σ(β * ((log πθ(c) - log πθ(r)) - (log πref(c) - log πref(r)))) ]
    """
    # chosen
    c_ids = batch["chosen_input"]["input_ids"]
    c_mask = batch["chosen_input"]["attention_mask"]

    # rejected
    r_ids = batch["rejected_input"]["input_ids"]
    r_mask = batch["rejected_input"]["attention_mask"]

    # log prob
    pi_c = sequence_log_prob(policy, c_ids, c_mask)
    pi_r = sequence_log_prob(policy, r_ids, r_mask)

    with torch.no_grad():
        ref_c = sequence_log_prob(reference, c_ids, c_mask)
        ref_r = sequence_log_prob(reference, r_ids, r_mask)

    # DPO 逻辑项
    logits = beta * ((pi_c - pi_r) - (ref_c - ref_r))  # [B]

    loss = -F.logsigmoid(logits).mean()
    return loss


def multi_objective_dpo_loss(
    policy: PreTrainedModel,
    reference: PreTrainedModel,
    batch: dict,
    beta: float = 0.1,
    weights: dict[str, float] | None = None,
    brevity_coef: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    多目标 DPO：L_total = Σ w_i * L_DPO^i
    - 目前支持的 objective:
      - "base": 标准 DPO
      - "brevity": 对 log p 加上长度惩罚（鼓励更短的输出）
    """
    if not weights:
        weights = {"base": 1.0}

    # 预计算基础 log prob 与长度
    c_ids = batch["chosen_input"]["input_ids"]
    c_mask = batch["chosen_input"]["attention_mask"]
    r_ids = batch["rejected_input"]["input_ids"]
    r_mask = batch["rejected_input"]["attention_mask"]

    pi_c_base = sequence_log_prob(policy, c_ids, c_mask)
    pi_r_base = sequence_log_prob(policy, r_ids, r_mask)
    with torch.no_grad():
        ref_c_base = sequence_log_prob(reference, c_ids, c_mask)
        ref_r_base = sequence_log_prob(reference, r_ids, r_mask)

    # 近似长度 = 有效 token 数（不含首 token 对齐差异影响较小）
    c_len = c_mask.sum(dim=-1).to(pi_c_base.dtype)
    r_len = r_mask.sum(dim=-1).to(pi_r_base.dtype)

    total_loss = None
    per_obj_stats: dict[str, float] = {}

    for name, w in weights.items():
        if name == "base":
            pi_c, pi_r = pi_c_base, pi_r_base
            ref_c, ref_r = ref_c_base, ref_r_base
        elif name == "brevity":
            # 对 policy/ref 的 log prob 加长度惩罚（鼓励短输出）
            pi_c = pi_c_base - brevity_coef * c_len
            pi_r = pi_r_base - brevity_coef * r_len
            ref_c = ref_c_base - brevity_coef * c_len
            ref_r = ref_r_base - brevity_coef * r_len
        else:
            # 未知目标，跳过
            continue

        logits = beta * ((pi_c - pi_r) - (ref_c - ref_r))
        loss = -F.logsigmoid(logits).mean()
        weighted = w * loss
        total_loss = weighted if total_loss is None else total_loss + weighted

        # 记录该 objective 的统计（用于监控）
        with torch.no_grad():
            acc = (pi_c - pi_r > 0).float().mean().item()
            # 隐式奖励差：beta*((pi_c-ref_c)-(pi_r-ref_r))
            margin = (beta * ((pi_c - ref_c) - (pi_r - ref_r))).mean().item()
        per_obj_stats[f"{name}_loss"] = loss.item()
        per_obj_stats[f"{name}_acc"] = acc
        per_obj_stats[f"{name}_margin"] = margin

    assert total_loss is not None
    return total_loss, per_obj_stats