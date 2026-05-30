"""1D-CNN + BiLSTM model for CSI action recognition."""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    """CNN-BiLSTM classifier.

    Expected input shape is [batch, time_steps, n_features]. The feature
    dimension can be 64 raw subcarriers, fewer kept subcarriers, or 4-node
    early-fusion features such as 256.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        cnn_channels: tuple[int, int, int] = (32, 64, 128),
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        classifier_dropout: float = 0.5,
        use_attention: bool = False,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        c1, c2, c3 = cnn_channels

        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, c1, kernel_size=5, padding=2),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c1, kernel_size=5, padding=2),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        self.lstm = nn.LSTM(
            input_size=c3,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out = lstm_hidden * 2
        self.attention = (
            nn.MultiheadAttention(
                embed_dim=lstm_out,
                num_heads=attention_heads,
                dropout=lstm_dropout,
                batch_first=True,
            )
            if use_attention
            else None
        )
        self.attention_norm = nn.LayerNorm(lstm_out) if use_attention else None

        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(lstm_out, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, features], got {tuple(x.shape)}")

        x = x.permute(0, 2, 1).contiguous()
        x = self.cnn(x)
        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.lstm(x)

        if self.attention is not None:
            attn_out, _ = self.attention(x, x, x, need_weights=False)
            x = self.attention_norm(x + attn_out)

        x = x.mean(dim=1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
