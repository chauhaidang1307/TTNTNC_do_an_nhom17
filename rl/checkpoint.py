"""Lưu và nạp checkpoint. Trọng số KHÔNG đưa lên GitHub (thư mục checkpoints/ nằm trong .gitignore); lưu lên Google Drive."""
from functools import lru_cache
from pathlib import Path

import torch

from rl.network import QNet

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def save_checkpoint(path, net, network_cfg, episode, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "network": dict(network_cfg),
                "episode": int(episode), "extra": dict(extra or {})}, path)


@lru_cache(maxsize=8)
def load_network(path, device="cpu"):
    # Trả về mạng ở chế độ eval, đã nạp trọng số. Có cache theo (path, device): cùng một mạng dùng chung cho
    # nhiều tác tử trong một tiến trình (chỉ đọc).
    data = torch.load(path, map_location=device)
    net = QNet(**data["network"]).to(device)
    net.load_state_dict(data["state_dict"])
    net.eval()
    return net
