from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class CNNLSTM(nn.Module):
    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            feat_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            self.cnn = backbone
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),
        )

    def freeze_backbone(self) -> None:
        for p in self.cnn.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.cnn.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        f = self.cnn(x)
        f = f.view(b, t, -1)
        o, _ = self.lstm(f)
        last = o[:, -1, :]
        logit = self.head(last).squeeze(1)
        return logit
