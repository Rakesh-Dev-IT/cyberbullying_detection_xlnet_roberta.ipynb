import torch
from torch.utils.data import Dataset

class CyberDataset(Dataset):
    def __init__(self, texts, labels, xlnet_tokenizer, roberta_tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.xlnet_tokenizer = xlnet_tokenizer
        self.roberta_tokenizer = roberta_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        xlnet_enc = self.xlnet_tokenizer(
            text, padding="max_length",
            truncation=True, max_length=self.max_len,
            return_tensors="pt"
        )

        roberta_enc = self.roberta_tokenizer(
            text, padding="max_length",
            truncation=True, max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "xlnet_input_ids": xlnet_enc["input_ids"].squeeze(),
            "xlnet_attention_mask": xlnet_enc["attention_mask"].squeeze(),
            "roberta_input_ids": roberta_enc["input_ids"].squeeze(),
            "roberta_attention_mask": roberta_enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
