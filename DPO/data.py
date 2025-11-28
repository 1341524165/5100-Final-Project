import json
import random
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


class PreferenceDataset(Dataset):
    """
    每条样本：
      {
        "prompt": "...",
        "chosen": "...",
        "rejected": "..."
      }
    """
    def __init__(self,
                 data: List[Dict],
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 512,
                 noise_rate: float = 0.0):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.noise_rate = noise_rate

    def _encode(self, prompt: str, answer: str):
        text = prompt + answer
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        # [1, L] -> [L]
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return encoded

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # Noise Injection: Randomly flip chosen and rejected
        if self.noise_rate > 0 and random.random() < self.noise_rate:
            chosen, rejected = rejected, chosen

        return {
            "chosen_input": self._encode(prompt, chosen),
            "rejected_input": self._encode(prompt, rejected),
        }

    def __len__(self):
        return len(self.data)


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def build_dataloaders(tokenizer: PreTrainedTokenizer,
                      train_path: str,
                      val_path: str | None,
                      max_length: int,
                      batch_size: int,
                      noise_rate: float = 0.0):
    train_data = load_jsonl(train_path)
    # Only inject noise into training data
    train_dataset = PreferenceDataset(train_data, tokenizer, max_length, noise_rate=noise_rate)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = None
    if val_path is not None:
        val_data = load_jsonl(val_path)
        val_dataset = PreferenceDataset(val_data, tokenizer, max_length)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

    return train_loader, val_loader