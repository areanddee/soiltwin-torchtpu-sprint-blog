"""CO2-flux LSTM with temporal attention, a colonization detector and a scale-shift head.

Input: (batch, 2016 half-hour steps, 6 channels). Output: (batch, steps, 1) CO2 flux (normalized).
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Per-time-step gate computed from the LSTM state and the raw inputs, smoothed over ~1 day."""

    def __init__(self, hidden_size: int, attention_size: int = 32, input_size: int = 5):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1),
        )
        self.input_attention = nn.Sequential(
            nn.Linear(input_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1),
        )
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, lstm_hidden: torch.Tensor, inputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq, hidden = lstm_hidden.shape
        hidden_scores = self.attention_net(lstm_hidden).squeeze(-1)
        input_scores = self.input_attention(inputs).squeeze(-1)
        scores = (hidden_scores + input_scores) / (self.temperature + 1e-8)
        if mask is not None:
            scores = scores.masked_fill(mask < 0.5, float('-inf'))

        # Local smoothing (47 steps ~ one day of half-hours)
        kernel_size = min(48, seq)
        if kernel_size % 2 == 0:
            kernel_size -= 1
        kernel_size = max(1, kernel_size)
        padding = kernel_size // 2
        smoothed = F.avg_pool1d(scores.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=padding)
        smoothed = smoothed.squeeze(1)
        if smoothed.shape[1] != seq:
            smoothed = smoothed[:, :seq]

        attention = torch.sigmoid(smoothed)
        attended = lstm_hidden * attention.unsqueeze(-1)
        return attended, attention


class ColonizationDetector(nn.Module):
    """Bidirectional LSTM that predicts, per step, whether microbial colonization has started (supervised)."""

    def __init__(self, input_size: int = 5, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(inputs)
        return self.fc(lstm_out).squeeze(-1)


class CO2FluxLSTM(nn.Module):
    """2-layer LSTM -> (attention blended by colonization probability) -> MLP head -> per-trace scale and shift."""

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,
        attention_size: int = 32,
        use_scale_shift: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        self.use_scale_shift = use_scale_shift

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=False)
        if use_attention:
            self.attention = TemporalAttention(hidden_size, attention_size, input_size)
            self.colonization_detector = ColonizationDetector(input_size, 32)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        if use_scale_shift:
            self.scale_shift_head = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, 2),  # [log_scale, offset]
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attention: bool = False):
        lstm_out, (h_n, c_n) = self.lstm(x)
        attention_weights = None
        colonization_probs = None

        if self.use_attention:
            attended, attention_weights = self.attention(lstm_out, x, mask)
            colonization_probs = self.colonization_detector(x)
            alpha = colonization_probs.unsqueeze(-1)
            combined = alpha * attended + (1 - alpha) * lstm_out
            y_raw = self.fc(combined)
        else:
            y_raw = self.fc(lstm_out)

        if self.use_scale_shift:
            scale_offset = self.scale_shift_head(h_n[-1])
            scale = F.softplus(scale_offset[:, 0:1]) + 0.5
            offset = scale_offset[:, 1:2]
            y = scale.unsqueeze(1) * y_raw + offset.unsqueeze(1)
        else:
            y = y_raw

        if return_attention:
            return y, attention_weights, colonization_probs
        return y

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
