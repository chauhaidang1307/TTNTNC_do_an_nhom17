"""Mạng Q: nhận tensor 2 x 6 x 7 (góc nhìn người đến lượt), trả về 7 giá trị Q (một cho mỗi cột).

Q(s, a) = giá trị kỳ vọng (trong [-1, 1]) của việc thả quân vào cột a, theo góc nhìn người đang đi.
3 lớp tích chập (mặc định 64 kênh) rồi 2 lớp kết nối đầy đủ (theo kế hoạch đã nộp).
"""
from torch import nn

from engine.board import COLS, ROWS


class QNet(nn.Module):
    def __init__(self, channels=64, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * ROWS * COLS, hidden), nn.ReLU(),
            nn.Linear(hidden, COLS),
        )

    def forward(self, x):
        return self.fc(self.conv(x))
