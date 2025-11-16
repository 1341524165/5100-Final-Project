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