import random

from agents.base import Agent


class RandomAgent(Agent):
    # Cấp 1: chọn đều ngẫu nhiên trong các nước hợp lệ (baseline).
    name = "Random"

    def __init__(self, seed=None):
        self.seed = seed
        self.rng = random.Random(seed)  # bộ sinh riêng, để tái lập được theo seed

    def choose_action(self, board):
        actions = board.legal_actions()
        if not actions:
            raise ValueError("Không còn nước hợp lệ (ván đã kết thúc)")
        return self.rng.choice(actions)
