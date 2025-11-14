# 简化版DPO项目方案 - 基于TRL库

## 为什么改变策略？

**原方案问题**：
- ❌ 从零实现整个训练框架（~3500行代码）
- ❌ 对初学者过于复杂
- ❌ 容易在工程细节上浪费时间

**新方案优势**：
- ✅ 使用TRL库（HuggingFace官方DPO实现）
- ✅ 专注于理解核心算法
- ✅ 实现创新部分（噪声注入 + 多目标）
- ✅ 4周内可完成

---

## 新的项目结构

```
Final Project/
├── experiments/
│   ├── 1_baseline_dpo.py          # 使用TRL库训练baseline
│   ├── 2_noise_robustness.py      # 噪声鲁棒性实验
│   └── 3_multi_objective_dpo.py   # 多目标DPO
│
├── src/
│   ├── custom_loss.py             # 你自己实现的loss函数（核心）
│   ├── noise_injection.py         # 噪声注入工具
│   └── multi_objective.py         # 多目标扩展
│
├── analysis/
│   └── results_analysis.ipynb     # 结果分析
│
└── README.md
```

**只需要实现3个核心文件，总共约500行代码！**

---

## 实施步骤

### Week 1: Baseline复现（使用TRL库）

#### 第1步：安装TRL库
```bash
pip install trl transformers datasets torch accelerate
```

#### 第2步：Baseline DPO训练（使用TRL现成代码）

**experiments/1_baseline_dpo.py**（约80行）：

```python
"""
Baseline DPO训练 - 使用TRL库

目的：验证环境、理解DPO工作流程、建立性能基准
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
import torch

def main():
    # 1. 加载模型和tokenizer
    model_name = "gpt2"  # 小模型，快速测试
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载数据集
    dataset = load_dataset("Anthropic/hh-rlhf")

    # 3. 配置DPO训练
    training_args = DPOConfig(
        output_dir="./results/baseline",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        beta=0.1,  # DPO温度参数
        logging_steps=10,
    )

    # 4. 创建DPO训练器（TRL库提供）
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )

    # 5. 训练！
    trainer.train()

    # 6. 保存模型
    trainer.save_model("./results/baseline/final_model")

    print("✓ Baseline DPO训练完成！")

if __name__ == "__main__":
    main()
```

**就这么简单！TRL库帮你处理了所有复杂细节。**

---

### Week 1-2: 理解并修改核心Loss

#### 第3步：实现自己的Loss函数（证明理解算法）

**src/custom_loss.py**（约150行）：

```python
"""
自定义DPO loss实现

目的：
1. 深入理解DPO算法
2. 为噪声注入和多目标扩展打基础
"""

import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps,      # 策略模型对chosen的log概率
    policy_rejected_logps,    # 策略模型对rejected的log概率
    reference_chosen_logps,   # 参考模型对chosen的log概率
    reference_rejected_logps, # 参考模型对rejected的log概率
    beta=0.1,                 # 温度参数
):
    """
    标准DPO loss实现

    公式（来自Rafailov et al. 2023, Equation 7）：
    L_DPO = -E[log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x)
                          - log π_θ(y_l|x) + log π_ref(y_l|x)))]

    直观理解：
    - 我们希望策略模型π_θ在chosen上的概率 > rejected上的概率
    - 同时不要偏离参考模型π_ref太远（由β控制）
    """

    # 计算log ratio
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps

    # DPO loss核心公式
    logits = beta * (policy_logratios - reference_logratios)
    loss = -F.logsigmoid(logits).mean()

    # 计算统计指标（用于监控）
    with torch.no_grad():
        # 隐式奖励
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)

        # 准确率：模型多少次更偏好chosen？
        accuracy = (policy_logratios > 0).float().mean()

    return loss, {
        "loss": loss.item(),
        "accuracy": accuracy.item(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
    }


def noisy_dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta=0.1,
    is_flipped=None,  # 标记哪些样本被翻转了
):
    """
    带噪声的DPO loss（你的创新1）

    思路：
    - 对于被翻转的样本，chosen和rejected实际上是反的
    - 我们需要考虑这个噪声对训练的影响
    """

    # 对于被翻转的样本，交换chosen和rejected
    if is_flipped is not None:
        # 创建副本避免修改原数据
        policy_chosen = policy_chosen_logps.clone()
        policy_rejected = policy_rejected_logps.clone()
        reference_chosen = reference_chosen_logps.clone()
        reference_rejected = reference_rejected_logps.clone()

        # 翻转标记为True的样本
        policy_chosen[is_flipped], policy_rejected[is_flipped] = \
            policy_rejected[is_flipped], policy_chosen[is_flipped]
        reference_chosen[is_flipped], reference_rejected[is_flipped] = \
            reference_rejected[is_flipped], reference_chosen[is_flipped]
    else:
        policy_chosen = policy_chosen_logps
        policy_rejected = policy_rejected_logps
        reference_chosen = reference_chosen_logps
        reference_rejected = reference_rejected_logps

    # 使用标准DPO loss
    return dpo_loss(
        policy_chosen, policy_rejected,
        reference_chosen, reference_rejected,
        beta
    )


def multi_objective_dpo_loss(
    policy_chosen_logps_dict,      # {"obj1": tensor, "obj2": tensor}
    policy_rejected_logps_dict,
    reference_chosen_logps_dict,
    reference_rejected_logps_dict,
    weights,                        # {"obj1": 0.6, "obj2": 0.4}
    beta=0.1,
):
    """
    多目标DPO loss（你的创新2）

    公式：L_MO-DPO = Σ w_i * L_DPO^i

    例子：
    - obj1 = informativeness（信息量）
    - obj2 = brevity（简洁性）
    - weights = {obj1: 0.7, obj2: 0.3}  # 更重视信息量
    """

    total_loss = 0.0
    all_stats = {}

    for obj_name in weights.keys():
        # 计算每个目标的DPO loss
        loss, stats = dpo_loss(
            policy_chosen_logps_dict[obj_name],
            policy_rejected_logps_dict[obj_name],
            reference_chosen_logps_dict[obj_name],
            reference_rejected_logps_dict[obj_name],
            beta=beta,
        )

        # 加权累加
        weighted_loss = weights[obj_name] * loss
        total_loss += weighted_loss

        # 记录每个目标的统计
        for key, value in stats.items():
            all_stats[f"{obj_name}_{key}"] = value

    all_stats["total_loss"] = total_loss.item()

    return total_loss, all_stats
```

**这个文件是你理解和创新的核心！**

---

### Week 2: 噪声鲁棒性实验

#### 第4步：噪声注入实验

**experiments/2_noise_robustness.py**（约100行）：

```python
"""
噪声鲁棒性实验

目的：测试DPO在不同噪声水平下的性能退化
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
import random

def inject_label_noise(dataset, noise_rate=0.1):
    """
    随机翻转noise_rate比例的标签

    Args:
        dataset: HuggingFace dataset
        noise_rate: 翻转比例（0.0-1.0）

    Returns:
        noisy_dataset: 带噪声的数据集
    """
    def flip_labels(example, idx):
        if random.random() < noise_rate:
            # 翻转chosen和rejected
            example['chosen'], example['rejected'] = \
                example['rejected'], example['chosen']
            example['is_flipped'] = True
        else:
            example['is_flipped'] = False
        return example

    noisy_dataset = dataset.map(
        flip_labels,
        with_indices=True,
        desc=f"Injecting {noise_rate*100:.0f}% noise"
    )

    return noisy_dataset


def train_with_noise(noise_level):
    """训练带噪声的DPO模型"""

    # 加载数据并注入噪声
    dataset = load_dataset("Anthropic/hh-rlhf")
    noisy_train = inject_label_noise(dataset["train"], noise_rate=noise_level)

    # 模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 训练配置
    training_args = DPOConfig(
        output_dir=f"./results/noise_{int(noise_level*100)}",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        beta=0.1,
    )

    # 训练
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=noisy_train,
        eval_dataset=dataset["test"],  # 用干净数据评估
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(f"./results/noise_{int(noise_level*100)}/final")

    print(f"✓ 噪声 {noise_level*100:.0f}% 实验完成")


if __name__ == "__main__":
    # 测试不同噪声水平
    for noise in [0.0, 0.05, 0.10, 0.20]:
        print(f"\n{'='*50}")
        print(f"训练噪声水平: {noise*100:.0f}%")
        print(f"{'='*50}")
        train_with_noise(noise)
```

---

### Week 3: 多目标DPO实验

**experiments/3_multi_objective_dpo.py**（约150行）

基于你的custom loss实现多目标训练，通过继承DPOTrainer并重写loss计算。

---

## 总代码量对比

| 方案 | 代码量 | 复杂度 | 时间 |
|------|--------|--------|------|
| **原方案（从零实现）** | ~3500行 | 非常高 | 6-8周 |
| **新方案（基于TRL）** | ~500行 | 中等 | 3-4周 |

---

## 在报告中如何表述

### ✅ 推荐写法：

```markdown
## Implementation

We implemented our DPO experiments using the HuggingFace TRL
library as our baseline training framework. To enable our
extensions for noise robustness and multi-objective optimization,
we re-implemented the core DPO loss function in `src/custom_loss.py`
based on Equation 7 from Rafailov et al. (2023).

This modular design allows us to:
1. Systematically inject label noise and study performance degradation
2. Extend DPO to multiple reward objectives with controllable weights
3. Focus on algorithmic innovation rather than engineering infrastructure
```

### ❌ 不要写：
"We implemented DPO from scratch..."（这听起来像过度承诺）

---

## 你现在应该怎么做？

### 1. **删除我之前的复杂实现**（可选）
```bash
# 或者重命名为backup
mv src src_backup
mv experiments experiments_backup
```

### 2. **安装TRL库**
```bash
pip install trl transformers datasets accelerate
```

### 3. **从最简单的开始**

我帮你创建一个超简单的测试脚本：
