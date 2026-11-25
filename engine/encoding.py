"""Biến bàn cờ thành đầu vào cho mạng neural (dùng cho RL và cấp 5).

Quy ước: tensor 2 x ROWS x COLS theo GÓC NHÌN CỦA NGƯỜI ĐẾN LƯỢT:
  kênh 0 = quân của người đang đến lượt, kênh 1 = quân của đối thủ.
Hàng 0 là hàng dưới cùng (giống Board.grid). Nhờ góc nhìn này một mạng dùng được cho cả hai bên.
"""
import numpy as np

from engine.board import COLS, EMPTY


def encode(board):
    g = np.array(board.grid, dtype=np.int8)
    mine = g == board.to_move
    theirs = (g != EMPTY) & ~mine
    return np.stack([mine, theirs]).astype(np.uint8)


def legal_mask(board):
    # Mảng bool độ dài COLS: True ở cột còn chỗ. Ván đã kết thúc thì toàn False.
    mask = np.zeros(COLS, dtype=bool)
    for col in board.legal_actions():
        mask[col] = True
    return mask
