"""Chạy thử (pilot): Heuristic đấu Random, chia đều đi trước / đi sau.

Chạy từ thư mục gốc:  python -m arena.heuristic_vs_random [--games 200] [--start-seed 0]
Ghi log thô: logs/heuristic_vs_random.csv (mỗi dòng một ván) và logs/heuristic_vs_random_meta.json.
Đây là lần chạy thử, chưa phải giải đấu chính thức: chưa có dải seed dev/test tách riêng.
"""
import argparse
import csv
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

from agents.heuristic import HeuristicAgent
from agents.random_agent import RandomAgent
from engine.board import Board, PLAYER_1, PLAYER_2

LOGS = Path(__file__).resolve().parent.parent / "logs"
SEED_OFFSET = 1_000_000  # seed của Random lệch khỏi seed của Heuristic để hai bên không dùng chung seed


def wilson(successes, n, z=1.96):
    # Khoảng tin cậy Wilson 95% cho một tỷ lệ.
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def play_one(heuristic_first, seed_h, seed_r):
    # Trả về (người thắng: "Heuristic"/"Random"/None, số nước, danh sách thời gian mỗi nước của Heuristic).
    h, r = HeuristicAgent(seed=seed_h), RandomAgent(seed=seed_r)
    agents = {PLAYER_1: h, PLAYER_2: r} if heuristic_first else {PLAYER_1: r, PLAYER_2: h}
    board = Board()
    times = []
    while not board.is_terminal():
        agent = agents[board.to_move]
        t0 = time.perf_counter()
        col = agent.choose_action(board)
        dt = time.perf_counter() - t0
        if agent is h:
            times.append(dt)
        board.step(col)
    winner = None if board.winner is None else agents[board.winner].name
    return winner, board.moves_made, times


def percentile(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, math.ceil(q * len(s)) - 1)]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200, help="tổng số ván (nên chẵn để chia đều đi trước/sau)")
    ap.add_argument("--start-seed", type=int, default=0)
    args = ap.parse_args()
    if args.games % 2:
        sys.exit("--games phải là số chẵn")

    LOGS.mkdir(exist_ok=True)
    rows, all_times = [], []
    for i in range(args.games):
        heuristic_first = i % 2 == 0
        seed_h = args.start_seed + i
        seed_r = seed_h + SEED_OFFSET
        winner, moves, times = play_one(heuristic_first, seed_h, seed_r)
        all_times += times
        rows.append(dict(game=i, heuristic_first=int(heuristic_first), seed_heuristic=seed_h, seed_random=seed_r,
                         winner=winner or "draw", moves=moves, heuristic_mean_ms=round(1000 * statistics.mean(times), 4)))

    with open(LOGS / "heuristic_vs_random.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def tally(sub):
        return (sum(r["winner"] == "Heuristic" for r in sub), sum(r["winner"] == "draw" for r in sub),
                sum(r["winner"] == "Random" for r in sub))

    summary = {}
    for label, sub in [("tổng", rows), ("Heuristic đi trước", [r for r in rows if r["heuristic_first"]]),
                       ("Heuristic đi sau", [r for r in rows if not r["heuristic_first"]])]:
        win, draw, loss = tally(sub)
        n = len(sub)
        score = (win + 0.5 * draw) / n
        lo, hi = wilson(win + 0.5 * draw, n)  # xấp xỉ: coi hòa là nửa thắng khi tính Wilson
        summary[label] = dict(games=n, win=win, draw=draw, loss=loss, score=round(score, 4),
                              wilson_low=round(lo, 4), wilson_high=round(hi, 4),
                              mean_moves=round(statistics.mean(r["moves"] for r in sub), 2))
        print(f"{label}: {n} ván | thắng {win} hòa {draw} thua {loss} | score {score:.3f} "
              f"[Wilson {lo:.3f}; {hi:.3f}] | trung bình {summary[label]['mean_moves']} nước")

    timing = dict(mean_ms=round(1000 * statistics.mean(all_times), 4),
                  p95_ms=round(1000 * percentile(all_times, 0.95), 4), moves_timed=len(all_times))
    print(f"Thời gian mỗi nước của Heuristic: trung bình {timing['mean_ms']} ms, p95 {timing['p95_ms']} ms ({timing['moves_timed']} nước)")

    meta = dict(games=args.games, start_seed=args.start_seed, seed_offset=SEED_OFFSET, summary=summary, timing=timing,
                python=platform.python_version(), machine=platform.platform(), processor=platform.processor(),
                note="Chạy thử (pilot), không phải giải đấu chính thức. Thời gian dao động theo máy.")
    (LOGS / "heuristic_vs_random_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
