"""
实验1：基线DPO训练

目的：
- 在Anthropic/hh-rlhf数据集上复现DPO结果
- 建立性能基准
- 验证训练流程正常工作

TODO: 基于simple_start.py，实现完整的基线训练
参考SIMPLIFIED_APPROACH.md中的详细示例
"""

# TODO: 实现基线DPO训练

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

    print("✓ Baseline DPO training completed! ")

if __name__ == "__main__":
    main()