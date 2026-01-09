import torch
import pandas as pd
import torch.nn as nn

from torch.utils.data import DataLoader
from transformers import (
    XLNetTokenizer, RobertaTokenizer,
    AdamW, get_cosine_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from dataset import CyberDataset
from model import HybridXLNetRoBERTa

# Load dataset
data = pd.read_csv(
    "/kaggle/input/cleaned-cyberbullying-dataset-csv/cleaned_cyberbullying_dataset.csv"
)

le = LabelEncoder()
data["label"] = le.fit_transform(data["label"])

train_texts, val_texts, train_labels, val_labels = train_test_split(
    data["text"].tolist(),
    data["label"].tolist(),
    test_size=0.1,
    stratify=data["label"],
    random_state=42
)

# Tokenizers
xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Datasets
train_dataset = CyberDataset(train_texts, train_labels, xlnet_tokenizer, roberta_tokenizer)
val_dataset = CyberDataset(val_texts, val_labels, xlnet_tokenizer, roberta_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridXLNetRoBERTa(num_classes=len(le.classes_)).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

epochs = 3
total_steps = len(train_loader) * epochs

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        outputs = model(
            batch["xlnet_input_ids"].to(device),
            batch["xlnet_attention_mask"].to(device),
            batch["roberta_input_ids"].to(device),
            batch["roberta_attention_mask"].to(device)
        )

        loss = criterion(outputs, batch["labels"].to(device))
        loss.backward()

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

# Save model & tokenizers
torch.save(model.state_dict(), "hybrid_xlnet_roberta_model.pth")
xlnet_tokenizer.save_pretrained("./xlnet_tokenizer")
roberta_tokenizer.save_pretrained("./roberta_tokenizer")
