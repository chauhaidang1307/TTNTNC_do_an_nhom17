"""Bộ nhớ kinh nghiệm (experience replay) dạng vòng, lưu bằng mảng numpy.

Mỗi bước chuyển: (s, a, r, s', done, mask của s'). s và s' đều theo góc nhìn người đến lượt tại chính trạng thái đó,
nên s' là góc nhìn của ĐỐI THỦ của người vừa đi. Tăng dữ liệu bằng lật gương (đảo cột) được áp dụng LÚC LẤY MẪU.
"""
import numpy as np
import torch

from engine.board import COLS, ROWS


class ReplayBuffer:
    def __init__(self, capacity, seed=None):
        self.capacity = capacity
        self.size = 0
        self.pos = 0
        self.states = np.zeros((capacity, 2, ROWS, COLS), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, 2, ROWS, COLS), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=bool)
        self.next_masks = np.zeros((capacity, COLS), dtype=bool)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def add(self, state, action, reward, next_state, done, next_mask):
        i = self.pos
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done
        self.next_masks[i] = next_mask
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device, flip=None):
        # flip: mảng bool độ dài batch_size (True = lật gương mẫu đó); None = mỗi mẫu lật với xác suất 1/2.
        idx = self.rng.integers(0, self.size, batch_size)
        if flip is None:
            flip = self.rng.random(batch_size) < 0.5
        flip = np.asarray(flip, dtype=bool)
        s, ns = self.states[idx], self.next_states[idx]  # lập chỉ mục mảng cho bản sao, an toàn để sửa
        a, nm = self.actions[idx], self.next_masks[idx]
        s[flip] = s[flip][..., ::-1]
        ns[flip] = ns[flip][..., ::-1]
        a[flip] = COLS - 1 - a[flip]
        nm[flip] = nm[flip][:, ::-1]
        return (
            torch.from_numpy(s).float().to(device),
            torch.from_numpy(a).to(device),
            torch.from_numpy(self.rewards[idx]).to(device),
            torch.from_numpy(ns).float().to(device),
            torch.from_numpy(self.dones[idx]).float().to(device),
            torch.from_numpy(nm).to(device),
        )
