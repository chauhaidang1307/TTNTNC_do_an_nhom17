"""Đo RAM (và VRAM nếu có) của từng tác tử.
Các số đo (đơn vị MB):
    rss_import   RAM của tiến trình sau khi import (đã gồm Python, numpy, torch...; giống nhau cho mọi tác tử)
    rss_build    RAM ngay sau khi tạo tác tử
    peak_wset    RAM đỉnh trong cả tiến trình (psutil, chỉ có trên Windows)
    delta_build  rss_build - rss_import (chi phí dựng tác tử, ví dụ nạp mạng)
    delta_peak   peak_wset - rss_import (chi phí đỉnh khi chơi, ví dụ cây MCTS)
    vram_alloc   torch.cuda.max_memory_allocated (chỉ tensor; chưa tính ngữ cảnh CUDA), chỉ khi device = cuda
"""
import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

import psutil

LOGS = Path(__file__).resolve().parent.parent / "logs"
DEFAULT_SPECS = ["random", "heuristic", "ab-d4", "ab-t50", "mcts-s1000", "mcts-t50", "dqn_RL-004_last"]
MB = 1024 * 1024
FIELDS = ["spec", "device", "games", "rss_import_mb", "rss_build_mb", "peak_wset_mb",
          "delta_build_mb", "delta_peak_mb", "vram_alloc_mb"]


def child(spec, games, device):
    # Chạy trong tiến trình con: dựng tác tử, chơi vài ván, in một dòng JSON.
    from agents.dqn_agent import DQNAgent
    from agents.heuristic import HeuristicAgent
    from arena.match import play_game
    from arena.tournament import make_agent

    proc = psutil.Process()
    rss_import = proc.memory_info().rss
    if spec.startswith("dqn_"):
        make = lambda seed: DQNAgent.from_spec(spec, seed=seed, device=device)
    else:
        make = lambda seed: make_agent(spec, seed)
    agent = make(0)
    rss_build = proc.memory_info().rss
    for i in range(games):
        a = agent if i == 0 else make(i)  # tạo tác tử mới mỗi ván giống arena
        h = HeuristicAgent(seed=i + 10_000_000)
        play_game(a, h) if i % 2 == 0 else play_game(h, a)
    peak = proc.memory_info().peak_wset
    vram = None
    if device == "cuda":
        import torch
        vram = torch.cuda.max_memory_allocated() / MB
    print("RESULT " + json.dumps(dict(
        spec=spec, device=device, games=games,
        rss_import_mb=round(rss_import / MB, 1), rss_build_mb=round(rss_build / MB, 1),
        peak_wset_mb=round(peak / MB, 1),
        delta_build_mb=round((rss_build - rss_import) / MB, 1),
        delta_peak_mb=round((peak - rss_import) / MB, 1),
        vram_alloc_mb=None if vram is None else round(vram, 2))))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Đo RAM/VRAM từng tác tử")
    ap.add_argument("--exp", help="mã thí nghiệm, ví dụ EXP-011")
    ap.add_argument("--specs", nargs="+", default=DEFAULT_SPECS)
    ap.add_argument("--games", type=int, default=10, help="số ván mỗi tác tử (chơi với Heuristic)")
    ap.add_argument("--no-cuda-row", action="store_true", help="không đo thêm dòng DQN chạy trên GPU")
    ap.add_argument("--child", help="(nội bộ) tên tác tử cho tiến trình con")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.child:
        child(args.child, args.games, args.device)
        return
    if not args.exp:
        sys.exit("Cần --exp")
    out_dir = LOGS / args.exp
    csv_path, meta_path = out_dir / f"{args.exp}_resources.csv", out_dir / f"{args.exp}_resources_meta.json"
    if csv_path.exists() or meta_path.exists():
        sys.exit(f"Không ghi đè log cũ: {csv_path.name}. Dùng mã thí nghiệm mới.")

    jobs = [(s, "cpu") for s in args.specs]
    if not args.no_cuda_row:
        jobs += [(s, "cuda") for s in args.specs if s.startswith("dqn_")]
    rows = []
    for spec, device in jobs:
        cmd = [sys.executable, "-m", "arena.resources", "--child", spec, "--games", str(args.games), "--device", device]
        p = subprocess.run(cmd, cwd=LOGS.parent, capture_output=True, text=True, encoding="utf-8")
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if p.returncode != 0 or not line:
            print(f"{spec} ({device}) LỖI:\n{p.stderr[-800:]}")
            continue
        r = json.loads(line[-1][7:])
        rows.append(r)
        print(f"{spec:22s} {device:4s} build +{r['delta_build_mb']:7.1f} MB | đỉnh +{r['delta_peak_mb']:7.1f} MB "
              f"| peak {r['peak_wset_mb']:7.1f} MB | VRAM {r['vram_alloc_mb']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    meta = dict(exp=args.exp, games_per_agent=args.games, opponent="heuristic", seeds="dev (0..), không tính điểm",
                python=platform.python_version(), machine=platform.platform(), processor=platform.processor(),
                total_ram_gb=round(psutil.virtual_memory().total / 1024 ** 3, 1),
                note="RAM đo bằng psutil (rss, peak_wset) trong tiến trình con riêng cho từng tác tử; "
                     "rss_import giống nhau cho mọi tác tử vì đã nạp torch. VRAM là max_memory_allocated "
                     "(chưa gồm ngữ cảnh CUDA). Arena chính thức chạy DQN trên CPU.")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Đã ghi:", csv_path.name, meta_path.name)


if __name__ == "__main__":
    main()
