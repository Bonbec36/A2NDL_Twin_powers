# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class PainCNN(nn.Module):
    """
    Input:  (B, F, T)  — B: batch size, F: feature channels (in_channels), T: time length
    Output: (B, C)     — C: number of classes (num_classes)

    Architecture
    ------------
    Conv1d Block × 4  ->  Global Average Pooling (over time)  ->  MLP head
    where each block = Conv1d + BatchNorm1d + ReLU + MaxPool1d

    Important: Any squeeze/mean is applied ONLY on the time dimension (dim = -1),
    never on the batch dimension.
    """
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()

        widths = [64, 128, 192]#[64, 128, 192]#, 192, 256]#]  # channel progression per block
        #widths = widths[::-1] #inverse the shape
        ks = 5                        # conv kernel size
        mp = 2                        # maxpool kernel/stride

        layers = []
        c_in = in_channels
        for c_out in widths:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=ks, padding=ks // 2, bias=False),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=mp, stride=mp),
            ]
            c_in = c_out

        # Feature extractor over time
        self.feat = nn.Sequential(*layers)

        # Global average pooling ONLY along time dimension
        self.gap = nn.AdaptiveAvgPool1d(1)  # (B, C, T') -> (B, C, 1)

        # Classifier head
        hidden = 64#default : 128
        self.head = nn.Sequential(
            nn.Linear(widths[-1], hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_classes),
        )

        # He/Kaiming for conv, ones/zeros for BN, Xavier for linear
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Expects x with shape (B, F, T). If the caller accidentally passes (B, T, F),
        the outer training/eval scripts should permute it before calling this method.
        """
        assert x.dim() == 3, f"Expected 3D (B,F,T), got {tuple(x.shape)}"
        B0 = x.size(0)  # remember incoming batch size

        # Feature extraction: (B, F, T) -> (B, C, T')
        x = self.feat(x)
        assert x.size(0) == B0, f"[feat] batch collapsed: {B0} -> {x.size(0)}"

        # GAP over time: (B, C, T') -> (B, C, 1)
        x = self.gap(x)
        assert x.size(0) == B0, f"[gap] batch collapsed: {B0} -> {x.size(0)}"

        # Remove the singleton time dim; squeeze only the last dim to avoid touching batch
        x = x.squeeze(-1)  # (B, C)
        assert x.size(0) == B0, f"[squeeze] batch collapsed: {B0} -> {x.size(0)}"

        # Classifier
        logits = self.head(x)  # (B, num_classes)
        assert logits.size(0) == B0, f"[head] batch collapsed: {B0} -> {logits.size(0)}"
        return logits
