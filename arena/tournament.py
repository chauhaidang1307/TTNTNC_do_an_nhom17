import argparse
import csv
import itertools
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

from agents.alphabeta import AlphaBetaAgent
from agents.heuristic import HeuristicAgent
from agents.random_agent import RandomAgent
from arena.match import play_game
from arena.metrics import percentile, score, wilson
from engine.board import PLAYER_1

LOGS = Path(__file__).resolve().parent.parent / "logs"
SPLIT_START = {"dev": 0, "test": 1_000_000}
SECOND_SEED_OFFSET = 10_000_000
GAME_FIELDS = ["exp", "split", "pair", "game", "agent_a", "agent_b", "a_first",
               "seed_a", "seed_b", "winner", "moves"]
MOVE_FIELDS = ["exp", "split", "pair", "game", "ply", "player", "agent", "col", "time_ms", "nodes", "depth"]


def make_agent(spec, seed):
    # Tạo tác tử từ tên (xem docstring đầu file).
    pruning = not spec.endswith("-nopr")
    base = spec if pruning else spec[:-5]
    if base == "random":
        return RandomAgent(seed=seed)
    if base == "heuristic":
        return HeuristicAgent(seed=seed)
    if base.startswith("ab-d"):
        return AlphaBetaAgent(depth=int(base[4:]), seed=seed, pruning=pruning)
    if base.startswith("ab-t"):
        return AlphaBetaAgent(depth=42, seed=seed, pruning=pruning, time_limit=int(base[4:]) / 1000)
    raise ValueError(f"Tên tác tử không hợp lệ: {spec}")


def run_pair(exp, split, spec_a, spec_b, games, game_rows, move_rows):
    # Ván i chẵn: A đi trước; ván i lẻ: B đi trước (chia đều). Mỗi ván dùng seed riêng, tạo agent mới.
    if games % 2:
        raise ValueError("games phải là số chẵn để chia đều đi trước/đi sau")
    pair = f"{spec_a}__vs__{spec_b}"
    start = SPLIT_START[split]
    for i in range(games):
        seed_a = start + i
        seed_b = start + i + SECOND_SEED_OFFSET
        a, b = make_agent(spec_a, seed_a), make_agent(spec_b, seed_b)
        a_first = i % 2 == 0
        winner, moves = play_game(a, b) if a_first else play_game(b, a)
        if winner is None:
            label = "draw"
        else:
            a_won = (winner == PLAYER_1) == a_first
            label = spec_a if a_won else spec_b
        game_rows.append(dict(exp=exp, split=split, pair=pair, game=i, agent_a=spec_a, agent_b=spec_b,
                              a_first=int(a_first), seed_a=seed_a, seed_b=seed_b, winner=label, moves=len(moves)))
        for m in moves:
            move_rows.append(dict(exp=exp, split=split, pair=pair, game=i, **m))


def summarize(rows, spec_a):
    # Tổng hợp một cặp theo góc nhìn của A.
    def tally(sub):
        w = sum(r["winner"] == spec_a for r in sub)
        d = sum(r["winner"] == "draw" for r in sub)
        n = len(sub)
        lo, hi = wilson(w + 0.5 * d, n)
        return dict(games=n, win=w, draw=d, loss=n - w - d, score=round(score(w, d, n), 4),
                    wilson_low=round(lo, 4), wilson_high=round(hi, 4))
    return dict(total=tally(rows),
                a_first=tally([r for r in rows if r["a_first"]]),
                a_second=tally([r for r in rows if not r["a_first"]]),
                mean_moves=round(statistics.mean(r["moves"] for r in rows), 2))


def timing_by_agent(move_rows):
    # Thời gian mỗi nước (ms), số nút/mô phỏng và độ sâu theo từng tên tác tử (gộp mọi ván).
    by = {}
    for m in move_rows:
        by.setdefault(m["agent"], []).append(m)
    out = {}
    for name, ms in by.items():
        t = [m["time_ms"] for m in ms]
        nodes = [m["nodes"] for m in ms if m["nodes"] != ""]
        depth = [m["depth"] for m in ms if m["depth"] != ""]
        out[name] = dict(moves=len(t), mean_ms=round(statistics.mean(t), 4), p95_ms=round(percentile(t, 0.95), 4),
                         mean_nodes=round(statistics.mean(nodes), 1) if nodes else None,
                         mean_depth=round(statistics.mean(depth), 2) if depth else None)
    return out


def pair_paths(exp, split, spec_a, spec_b):
    # Mỗi cặp một bộ ba file; tên có mã thí nghiệm, dải seed và tên hai agent,
    # ví dụ logs/EXP-002/heuristic_vs_ab-d3/EXP-002_dev_heuristic_vs_ab-d3_games.csv (kèm _moves.csv, _meta.json).
    # Cấu trúc: logs/<exp>/<A>_vs_<B>/<exp>_<split>_<A>_vs_<B>_{games.csv, moves.csv, meta.json}
    stem = f"{exp}_{split}_{spec_a}_vs_{spec_b}"
    folder = LOGS / exp / f"{spec_a}_vs_{spec_b}"
    return {k: folder / f"{stem}_{k}.{ext}" for k, ext in [("games", "csv"), ("moves", "csv"), ("meta", "json")]}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=LOGS.parent,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Giải đấu round-robin Connect Four")
    ap.add_argument("--exp", required=True, help="mã thí nghiệm, ví dụ EXP-002")
    ap.add_argument("--split", choices=list(SPLIT_START), default="dev")
    ap.add_argument("--games", type=int, default=100, help="số ván MỖI CẶP (chẵn)")
    ap.add_argument("--agents", nargs="+", required=True, help="tên các tác tử, xem docstring")
    ap.add_argument("--confirm-test", action="store_true", help="bắt buộc khi --split test")
    args = ap.parse_args()

    if args.split == "test" and not args.confirm_test:
        sys.exit("Dải test chỉ chạy một lần sau khi chốt cấu hình: thêm --confirm-test nếu chắc chắn.")
    if len(set(args.agents)) != len(args.agents) or len(args.agents) < 2:
        sys.exit("--agents cần ít nhất 2 tên khác nhau")
    if args.games % 2:
        sys.exit("--games phải là số chẵn")
    for spec in args.agents:  # báo lỗi tên sai trước khi chạy
        make_agent(spec, 0)

    pair_list = list(itertools.combinations(args.agents, 2))
    all_paths = {pair: pair_paths(args.exp, args.split, *pair) for pair in pair_list}
    existing = [p.name for paths in all_paths.values() for p in paths.values() if p.exists()]
    if existing:  # kiểm tra TRƯỚC khi chạy bất kỳ cặp nào
        sys.exit(f"Không ghi đè log cũ: {existing}. Dùng mã thí nghiệm mới.")

    for spec_a, spec_b in pair_list:
        game_rows, move_rows = [], []
        run_pair(args.exp, args.split, spec_a, spec_b, args.games, game_rows, move_rows)
        s = summarize(game_rows, spec_a)
        t = s["total"]
        print(f"{spec_a} vs {spec_b}: {t['win']}-{t['draw']}-{t['loss']} | score(A) {t['score']:.3f} "
              f"[Wilson {t['wilson_low']:.3f}; {t['wilson_high']:.3f}] | "
              f"A đi trước {s['a_first']['score']:.3f}, đi sau {s['a_second']['score']:.3f}")

        # Mỗi cặp một bộ ba file riêng, ghi ngay khi cặp đó xong (lỗi giữa chừng không mất cặp đã xong).
        paths = all_paths[(spec_a, spec_b)]
        paths["games"].parent.mkdir(parents=True, exist_ok=True)
        for key, fields, rows in [("games", GAME_FIELDS, game_rows), ("moves", MOVE_FIELDS, move_rows)]:
            with open(paths[key], "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
        meta = dict(exp=args.exp, split=args.split, games=args.games, agent_a=spec_a, agent_b=spec_b,
                    seed_start=SPLIT_START[args.split], second_seed_offset=SECOND_SEED_OFFSET,
                    summary=s, timing=timing_by_agent(move_rows),
                    git_commit=git_commit(), python=platform.python_version(),
                    machine=platform.platform(), processor=platform.processor(),
                    note="Thời gian dao động theo máy và tải; tác tử theo thời gian (ab-t) không tái lập chính xác, "
                         "xem cột depth trong file moves. RAM/VRAM chưa được đo. Điểm (score) theo góc nhìn của agent_a.")
        paths["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print("  Đã ghi:", ", ".join(p.name for p in paths.values()))


if __name__ == "__main__":
    main()
