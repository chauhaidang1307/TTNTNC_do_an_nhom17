import random

import numpy as np
import torch

from agents.base import Agent
from engine.encoding import encode, legal_mask
from rl.checkpoint import CHECKPOINT_DIR, load_network


def greedy_action(net, board, device="cpu"):
    # Cột hợp lệ có Q lớn nhất. Cột đầy bị loại bằng mặt nạ (không bao giờ được chọn).
    mask = legal_mask(board)
    x = torch.from_numpy(encode(board)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        q = net(x)[0].cpu().numpy()
    q[~mask] = -np.inf
    return int(np.argmax(q))


class DQNAgent(Agent):
    # Cấp 6: chọn nước theo mạng Q đã huấn luyện (greedy, có mặt nạ nước hợp lệ).
    # epsilon > 0 chỉ dùng khi huấn luyện hoặc muốn đa dạng hóa; đánh giá thì để 0.
    def __init__(self, network, device="cpu", epsilon=0.0, seed=None, name="DQN"):
        self.net = network
        self.device = device
        self.epsilon = epsilon
        self.seed = seed
        self.rng = random.Random(seed)
        self.name = name

    @classmethod
    def from_checkpoint(cls, path, device="cpu", epsilon=0.0, seed=None, name=None):
        net = load_network(str(path), device)
        return cls(net, device=device, epsilon=epsilon, seed=seed, name=name or f"DQN-{path.stem}")

    @classmethod
    def from_spec(cls, spec, seed=None, device="cpu"):
        # Tên dạng dqn_<run>_<checkpoint>, ví dụ dqn_RL-001_best -> checkpoints/RL-001/best.pt
        run, ckpt = spec[len("dqn_"):].rsplit("_", 1)
        path = CHECKPOINT_DIR / run / f"{ckpt}.pt"
        return cls.from_checkpoint(path, device=device, seed=seed, name=f"DQN-{run}-{ckpt}")

    def choose_action(self, board):
        legal = board.legal_actions()
        if not legal:
            raise ValueError("Không còn nước hợp lệ (ván đã kết thúc)")
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            return self.rng.choice(legal)
        return greedy_action(self.net, board, self.device)
