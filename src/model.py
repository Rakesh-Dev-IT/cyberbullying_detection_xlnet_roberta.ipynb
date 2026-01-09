import torch
import torch.nn as nn
from transformers import XLNetModel, RobertaModel

class HybridXLNetRoBERTa(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.xlnet = XLNetModel.from_pretrained("xlnet-base-cased")
        self.roberta = RobertaModel.from_pretrained("roberta-base")

        hidden_size = (
            self.xlnet.config.hidden_size +
            self.roberta.config.hidden_size
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, xlnet_ids, xlnet_mask, roberta_ids, roberta_mask):
        xlnet_out = self.xlnet(
            input_ids=xlnet_ids,
            attention_mask=xlnet_mask
        )

        roberta_out = self.roberta(
            input_ids=roberta_ids,
            attention_mask=roberta_mask
        )

        xlnet_pooled = xlnet_out.last_hidden_state[:, 0]
        roberta_pooled = roberta_out.last_hidden_state[:, 0]

        combined = torch.cat((xlnet_pooled, roberta_pooled), dim=1)
        return self.classifier(combined)
