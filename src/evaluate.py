import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from model import HybridXLNetRoBERTa

def evaluate(model, val_loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                batch["xlnet_input_ids"].to(device),
                batch["xlnet_attention_mask"].to(device),
                batch["roberta_input_ids"].to(device),
                batch["roberta_attention_mask"].to(device)
            )

            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            labels.extend(batch["labels"].numpy())

    cm = confusion_matrix(labels, preds)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.show()

    print(classification_report(labels, preds))
