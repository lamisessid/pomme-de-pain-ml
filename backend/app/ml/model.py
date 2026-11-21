"""
LSTM multi-tâches : ventes, pertes et classe de risque de gaspillage.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Attention sur la dimension temporelle, concaténée au dernier état caché."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = F.softmax(self.score(lstm_out).squeeze(-1), dim=1)          # (B, T)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)       # (B, H)
        return torch.cat([context, lstm_out[:, -1, :]], dim=1), weights      # (B, 2H)


class MultiTaskLSTM(nn.Module):
    """
    Entrée  : (batch, seq_len, input_size)
    Sorties : ventes (B,), pertes (B,) — normalisées par produit, positives —
              et logits de risque (B, 3).
    """

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.config = dict(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_size)
        ctx = 2 * hidden_size
        # Connexion résiduelle depuis les features du jour cible
        self.skip = nn.Linear(input_size, ctx)

        def head(out: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(ctx, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, out))

        self.sales_head = head(1)
        self.waste_head = head(1)
        self.risk_head = head(3)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        out, _ = self.lstm(x)
        context, weights = self.attention(out)
        context = context + self.skip(x[:, -1, :])
        sales = F.softplus(self.sales_head(context)).squeeze(-1)
        waste = F.softplus(self.waste_head(context)).squeeze(-1)
        risk_logits = self.risk_head(context)
        if return_attention:
            return sales, waste, risk_logits, weights
        return sales, waste, risk_logits
