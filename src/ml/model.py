"""
ChessStyleNet: a CNN policy network conditioned on a learned "style embedding".

Design:
  - Convolutional trunk over the (18, 8, 8) board tensor (ResNet-style, small).
  - A style embedding (per player_id during training) is broadcast and
    concatenated onto the trunk's spatial feature maps before the final
    policy convolution, so the same trunk produces different move
    distributions depending on *whose* style is selected.
  - Policy head outputs logits over the 4672 AlphaZero-style move classes.
  - Auxiliary value head (optional) predicts P(this move is a blunder),
    reusing the "blunder predictor" idea as a regularizer / bonus metric —
    trained on Lichess eval-based blunder labels if available.

At inference, two style embeddings can be linearly interpolated
(`style_a * alpha + style_b * (1 - alpha)`) to demo style blending.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.core.board_encoding import NUM_MOVE_CLASSES


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class HistoryEncoder(nn.Module):
    def __init__(self, num_features: int, hidden_size: int, style_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_size=num_features, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, style_dim)

    def forward(self, history_tensor):
        """
        history_tensor: (B, seq_len, num_features)
        returns: (B, style_dim)
        """
        _, hn = self.gru(history_tensor)
        return self.fc(hn[-1])  # taking the final hidden state


class ChessStyleNet(nn.Module):
    def __init__(
        self,
        board_channels: int = 18,
        trunk_channels: int = 64,
        num_res_blocks: int = 6,
        style_dim: int = 32,
        num_move_classes: int = NUM_MOVE_CLASSES,
        use_blunder_head: bool = True,
        hist_features: int = 3,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.history_encoder = HistoryEncoder(num_features=hist_features, hidden_size=64, style_dim=style_dim)

        self.stem = nn.Sequential(
            nn.Conv2d(board_channels, trunk_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(trunk_channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(trunk_channels) for _ in range(num_res_blocks)])

        # project style vector to a per-square feature map and concat onto trunk
        self.style_proj = nn.Linear(style_dim, trunk_channels)

        combined_channels = trunk_channels * 2  # trunk features + broadcast style features
        self.policy_conv = nn.Sequential(
            nn.Conv2d(combined_channels, trunk_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(trunk_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(trunk_channels, 73, 1),  # 73 move-planes per square
        )

        self.use_blunder_head = use_blunder_head
        if use_blunder_head:
            self.blunder_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(combined_channels, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )

    def style_vector(self, history_tensor: torch.Tensor = None, style_vec: torch.Tensor = None):
        """Get a (B, style_dim) style vector either from history_tensor or directly."""
        if style_vec is not None:
            return style_vec
        return self.history_encoder(history_tensor)

    def forward(self, board_tensor: torch.Tensor, history_tensor: torch.Tensor = None, style_vec: torch.Tensor = None):
        """
        board_tensor: (B, 18, 8, 8)
        history_tensor: (B, seq_len, num_features) float tensor
        style_vec:    (B, style_dim) float tensor, alternative to history_tensor
        returns: policy_logits (B, 4672), blunder_logit (B, 1) or None
        """
        b = board_tensor.shape[0]
        x = self.stem(board_tensor)
        x = self.res_blocks(x)  # (B, C, 8, 8)

        style = self.style_vector(history_tensor, style_vec)      # (B, style_dim)
        style_map = self.style_proj(style)                        # (B, C)
        style_map = style_map.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 8)  # (B, C, 8, 8)

        combined = torch.cat([x, style_map], dim=1)                # (B, 2C, 8, 8)

        policy = self.policy_conv(combined)                        # (B, 73, 8, 8)
        policy_logits = policy.permute(0, 2, 3, 1).reshape(b, -1)  # -> (B, 8*8*73) matching from_sq*73+plane

        blunder_logit = self.blunder_head(combined) if self.use_blunder_head else None
        return policy_logits, blunder_logit


if __name__ == "__main__":
    net = ChessStyleNet()
    board = torch.randn(4, 18, 8, 8)
    history = torch.randn(4, 10, 3) # 4 batches, 10 games, 3 features
    logits, blunder = net(board, history_tensor=history)
    print("policy_logits:", logits.shape, "blunder_logit:", blunder.shape)
    assert logits.shape == (4, NUM_MOVE_CLASSES)
    
    style_a = net.style_vector(history_tensor=torch.randn(1, 10, 3))
    style_b = net.style_vector(history_tensor=torch.randn(1, 10, 3))
    interp = 0.5 * style_a + 0.5 * style_b
    print("interpolated style vector:", interp.shape)
    print("Param count:", sum(p.numel() for p in net.parameters()))
