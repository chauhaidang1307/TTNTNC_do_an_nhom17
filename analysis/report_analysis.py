"""Phân tích sâu các log đã có để đưa vào báo cáo cuối kỳ. Mọi con số và hình đều sinh từ log, không nhập tay.

Chạy từ thư mục gốc:  python analysis/report_analysis.py
Đầu vào : logs/EXP-001/*/ (games.csv, moves.csv, meta.json), logs/heuristic_vs_random.csv,
          logs/RL-001/train_log.csv, logs/RL-002/train_log.csv
Đầu ra  : analysis/report_numbers.json và các hình trong analysis/figures/report/

Hai phép kiểm tra nhanh ở cuối (số nút Alpha-Beta bật/tắt cắt tỉa trên hai thế cờ cố định; tốc độ MCTS) CHỈ là kiểm tra
tham khảo chạy bằng chính script này, KHÔNG phải thí nghiệm chính thức có mã EXP; thời gian đo trong lúc máy còn chạy việc khác.
"""
import collections
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "analysis" / "figures" / "report"
EXP = ROOT / "logs" / "EXP-001"
NUMBERS = ROOT / "analysis" / "report_numbers.json"

from agents.alphabeta import WINDOWS, AlphaBetaAgent  # noqa: E402
from agents.mcts import MCTSAgent  # noqa: E402
from arena.metrics import wilson  # noqa: E402
from engine.board import COLS, PLAYER_1, ROWS, Board  # noqa: E402

BLUE, ORANGE, GREEN, PURPLE, GRAY, RED, YELLOW = "#2563eb", "#d97706", "#059669", "#7c3aed", "#6b7280", "#e53935", "#fdd835"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
R = {}  # toàn bộ số liệu xuất ra JSON


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_pairs():
    pairs = []
    for d in sorted(EXP.iterdir()):
        if not d.is_dir():
            continue
        meta = json.loads(next(d.glob("*_meta.json")).read_text(encoding="utf-8"))
        games = read_csv(next(d.glob("*_games.csv")))
        moves = read_csv(next(d.glob("*_moves.csv")))
        by_game = collections.defaultdict(list)
        for m in moves:
            by_game[int(m["game"])].append(m)
        for g in by_game.values():
            g.sort(key=lambda m: int(m["ply"]))
        pairs.append(dict(name=f"{meta['agent_a']} vs {meta['agent_b']}", a=meta["agent_a"], b=meta["agent_b"],
                          meta=meta, games=games, moves=moves, by_game=by_game))
    return pairs


def p1_won(game, a):
    # Người đi trước (PLAYER_1) có thắng ván này không? a_first=1: A đi trước.
    if game["winner"] == "draw":
        return None
    a_won = game["winner"] == a
    return a_won if int(game["a_first"]) else (not a_won)


# ---------- 1. độ dài ván ----------
def analysis_lengths(pairs):
    out = {}
    for p in pairs:
        L = [int(g["moves"]) for g in p["games"]]
        out[p["name"]] = dict(mean=round(statistics.mean(L), 2), median=statistics.median(L), min=min(L), max=max(L),
                              p95=sorted(L)[max(0, int(np.ceil(0.95 * len(L))) - 1)])
    R["game_lengths"] = out
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    data = [[int(g["moves"]) for g in p["games"]] for p in pairs]
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.55, medianprops=dict(color="black"))
    for b in bp["boxes"]:
        b.set(facecolor="#bfdbfe", edgecolor=BLUE)
    ax.set_yticklabels([p["name"] for p in pairs])
    ax.invert_yaxis()
    ax.set_xlabel("Số nước đi của một ván (tối đa 42)")
    ax.set_title("Độ dài ván theo cặp (100 ván mỗi cặp)", fontsize=10)
    ax.grid(axis="x", color="#e5e7eb")
    fig.tight_layout()
    fig.savefig(OUT / "fig_game_lengths.png", dpi=200)
    plt.close(fig)


# ---------- 2. lợi thế người đi trước ----------
def analysis_first_mover(pairs):
    out, tot = {}, dict(p1=0, p2=0, draw=0)
    for p in pairs:
        c = dict(p1=0, p2=0, draw=0)
        for g in p["games"]:
            w = p1_won(g, p["a"])
            c["draw" if w is None else ("p1" if w else "p2")] += 1
        out[p["name"]] = c
        for k in tot:
            tot[k] += c[k]
    n = sum(tot.values())
    lo, hi = wilson(tot["p1"] + 0.5 * tot["draw"], n)
    R["first_mover"] = dict(per_pair=out, total=tot, games=n, p1_score=round((tot["p1"] + 0.5 * tot["draw"]) / n, 4),
                            wilson=[round(lo, 4), round(hi, 4)])
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    names = [p["name"] for p in pairs]
    p1 = [out[x]["p1"] for x in names]
    dr = [out[x]["draw"] for x in names]
    p2 = [out[x]["p2"] for x in names]
    y = np.arange(len(names))
    ax.barh(y, p1, color=RED, label="người đi trước thắng")
    ax.barh(y, dr, left=p1, color=GRAY, label="hòa")
    ax.barh(y, p2, left=[a + b for a, b in zip(p1, dr)], color=YELLOW, label="người đi sau thắng")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Số ván (trên 100)")
    ax.set_title("Kết quả theo LƯỢT ĐI (không phân biệt tác tử nào)", fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_first_mover.png", dpi=200)
    plt.close(fig)


# ---------- 3. chi phí theo nước đi ----------
def analysis_ply_cost(pairs):
    buckets = [(1, 10), (11, 20), (21, 30), (31, 42)]
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: dict(ms=[], nodes=[], depth=[])))
    for p in pairs:
        for m in p["moves"]:
            if m["nodes"] == "":
                continue
            ply = int(m["ply"])
            for lo, hi in buckets:
                if lo <= ply <= hi:
                    d = acc[m["agent"]][(lo, hi)]
                    d["ms"].append(float(m["time_ms"]))
                    d["nodes"].append(float(m["nodes"]))
                    d["depth"].append(float(m["depth"]))
    out = {}
    for agent, bd in acc.items():
        out[agent] = {f"{lo}-{hi}": dict(moves=len(v["ms"]), mean_ms=round(statistics.mean(v["ms"]), 3),
                                         mean_nodes=round(statistics.mean(v["nodes"]), 1),
                                         mean_depth=round(statistics.mean(v["depth"]), 2)) for (lo, hi), v in sorted(bd.items())}
    R["ply_cost"] = out
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for agent, color in (("AB-d3", BLUE), ("AB-d4", ORANGE)):
        if agent not in out:
            continue
        keys = list(out[agent].keys())
        a1.plot(keys, [out[agent][k]["mean_nodes"] for k in keys], marker="o", color=color, label=agent)
        a2.plot(keys, [out[agent][k]["mean_ms"] for k in keys], marker="o", color=color, label=agent)
    a1.set_title("(a) Số nút trung bình mỗi nước", fontsize=10)
    a2.set_title("(b) Thời gian trung bình mỗi nước (ms)", fontsize=10)
    for ax in (a1, a2):
        ax.set_xlabel("Giai đoạn ván (số thứ tự nước đi)")
        ax.grid(color="#e5e7eb")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ply_cost.png", dpi=200)
    plt.close(fig)


# ---------- 4. mở đầu ----------
def analysis_openings(pairs):
    first = collections.defaultdict(collections.Counter)   # tác tử khi đi trước: cột của nước 1
    second = collections.defaultdict(collections.Counter)  # tác tử khi đi sau: cột của nước 2
    for p in pairs:
        for g in p["by_game"].values():
            first[g[0]["agent"]][int(g[0]["col"])] += 1
            if len(g) > 1:
                second[g[1]["agent"]][int(g[1]["col"])] += 1
    R["openings"] = dict(first_move={a: {str(c + 1): n for c, n in sorted(cnt.items())} for a, cnt in first.items()},
                         reply={a: {str(c + 1): n for c, n in sorted(cnt.items())} for a, cnt in second.items()})
    agents = ["Random", "Heuristic", "AB-d3", "AB-d4"]
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 3.8), sharey="row")
    for j, ag in enumerate(agents):
        for i, (src, title) in enumerate(((first, "nước 1 (đi trước)"), (second, "nước 2 (đi sau)"))):
            cnt = src.get(ag, collections.Counter())
            axes[i][j].bar(range(1, COLS + 1), [cnt.get(c, 0) for c in range(COLS)], color=BLUE if i == 0 else ORANGE)
            axes[i][j].set_title(f"{ag}: {title}", fontsize=8)
            axes[i][j].set_xticks(range(1, COLS + 1))
    axes[1][0].set_xlabel("cột (1 đến 7)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_openings.png", dpi=200)
    plt.close(fig)


# ---------- 5. bootstrap và độ nhạy khi loại ván trùng ----------
def analysis_robustness(pairs):
    rng = np.random.default_rng(0)
    out = {}
    for p in pairs:
        pts = np.array([{p["a"]: 1.0, "draw": 0.5}.get(g["winner"], 0.0) for g in p["games"]])
        boots = rng.choice(pts, size=(10000, len(pts)), replace=True).mean(axis=1)
        b_lo, b_hi = np.percentile(boots, [2.5, 97.5])
        sc = pts.mean()
        w_lo, w_hi = wilson(pts.sum(), len(pts))
        # loại các ván có cùng (lượt đi, chuỗi nước) giữ một bản
        seen, keep = set(), []
        for g in p["games"]:
            key = (g["a_first"], tuple(m["col"] for m in p["by_game"][int(g["game"])]))
            if key not in seen:
                seen.add(key)
                keep.append({p["a"]: 1.0, "draw": 0.5}.get(g["winner"], 0.0))
        u = np.array(keep)
        u_lo, u_hi = wilson(u.sum(), len(u))
        out[p["name"]] = dict(score=round(float(sc), 4), wilson=[round(w_lo, 4), round(w_hi, 4)],
                              bootstrap=[round(float(b_lo), 4), round(float(b_hi), 4)],
                              unique_games=len(u), unique_score=round(float(u.mean()), 4),
                              unique_wilson=[round(u_lo, 4), round(u_hi, 4)])
    R["robustness"] = out
    R["robustness_note"] = "bootstrap 10.000 lần, rng seed 0; ván trùng = cùng lượt đi và cùng chuỗi nước"


# ---------- 6. ván điển hình (chọn theo quy tắc cố định, không theo kết quả) ----------
def draw_board(ax, grid, win=None, last=None, title=None, fs=7.5):
    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), COLS, ROWS, boxstyle="round,pad=0,rounding_size=0.15", fc="#1f4fb0", ec="#123a8a"))
    for r in range(ROWS):
        for c in range(COLS):
            v = grid[r][c]
            color = "#f2f4f8" if v == 0 else (RED if v == 1 else YELLOW)
            ax.add_patch(Circle((c + 0.5, r + 0.5), 0.38, fc=color, ec="#123a8a", lw=0.6))
            if win and (r, c) in win:
                ax.add_patch(Circle((c + 0.5, r + 0.5), 0.40, fc="none", ec="black", lw=2.0))
            if last == (r, c):
                ax.add_patch(Circle((c + 0.5, r + 0.5), 0.09, fc="black"))
    if title:
        ax.set_title(title, fontsize=fs, pad=3)


def replay(cols):
    b = Board()
    for c in cols:
        b.step(int(c))
    return b


def winning_cells(b):
    if b.winner is None:
        return None
    from engine.board import DIRECTIONS, WIN_LENGTH
    for r in range(ROWS):
        for c in range(COLS):
            if b.grid[r][c] != b.winner:
                continue
            for dr, dc in DIRECTIONS:
                cells = [(r + dr * i, c + dc * i) for i in range(WIN_LENGTH)]
                if all(0 <= rr < ROWS and 0 <= cc < COLS and b.grid[rr][cc] == b.winner for rr, cc in cells):
                    return set(cells)
    return None


def analysis_typical(pairs):
    chosen = []
    for p in pairs:
        if p["name"] in ("ab-d3 vs ab-d4", "heuristic vs ab-d3", "heuristic vs ab-d4"):
            for gi in (0, 1):  # quy tắc: hai ván đầu của mỗi cặp (seed nhỏ nhất, A đi trước rồi B đi trước)
                chosen.append((p, gi, "hai ván đầu của cặp"))
    for p in pairs:
        if p["name"] == "ab-d3 vs ab-d4":  # quy tắc: ván hòa đầu tiên theo thứ tự ván
            for g in p["games"]:
                if g["winner"] == "draw":
                    chosen.append((p, int(g["game"]), "ván hòa đầu tiên"))
                    break
    info = []
    n = len(chosen)
    cols = 3
    rows_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(7.6, 2.75 * rows_ + 0.2))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (p, gi, why) in zip(axes, chosen):
        g = p["games"][gi]
        seq = [m["col"] for m in p["by_game"][gi]]
        b = replay(seq)
        last = None
        if b.history:
            c, r = b.history[-1]
            last = (r, c)
        first_name = p["a"] if int(g["a_first"]) else p["b"]
        res = "hòa" if g["winner"] == "draw" else f"{g['winner']} thắng"
        draw_board(ax, b.grid, winning_cells(b), last, f"{p['name']}, ván {gi}\n{first_name} đi trước\n{res}; {len(seq)} nước", fs=7.5)
        info.append(dict(pair=p["name"], game=gi, rule=why, first=first_name, result=g["winner"], plies=len(seq),
                         seed_a=int(g["seed_a"]), seed_b=int(g["seed_b"]), moves=[int(c) + 1 for c in seq]))
    fig.tight_layout()
    fig.savefig(OUT / "fig_typical_games.png", dpi=200)
    plt.close(fig)
    R["typical_games"] = info


# ---------- 7. số cửa sổ phủ mỗi ô ----------
def analysis_windows():
    cover = np.zeros((ROWS, COLS), dtype=int)
    for w in WINDOWS:
        for r, c in w:
            cover[r][c] += 1
    R["window_cover"] = cover[::-1].tolist()  # hàng trên cùng trước
    R["windows_total"] = len(WINDOWS)
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    ax.imshow(cover[::-1], cmap="Blues", vmin=0, vmax=cover.max() + 2)
    for r in range(ROWS):
        for c in range(COLS):
            ax.text(c, ROWS - 1 - r, str(cover[r][c]), ha="center", va="center", fontsize=10)
    ax.set_xticks(range(COLS))
    ax.set_xticklabels(range(1, COLS + 1))
    ax.set_yticks([])
    ax.set_xlabel("cột (1 đến 7)")
    ax.set_title("Số cửa sổ 4 ô phủ mỗi ô (tổng 69 cửa sổ)", fontsize=10)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_window_cover.png", dpi=200)
    plt.close(fig)



# ---------- 7b. cách thua: phát lại mọi ván phân thắng bại ----------
def winning_columns(board, player):
    # Các cột mà `player` thả vào là thắng ngay (không đổi bàn cờ).
    cols = []
    for c in board.legal_actions():
        b = board.clone()
        b.to_move = player  # xét như thể đến lượt player
        b.step(c)
        if b.winner == player:
            cols.append(c)
    return cols


def classify_loss(cols):
    # cols: chuỗi cột của một ván phân thắng bại. Xét nước của NGƯỜI THUA ngay trước nước thắng cuối cùng.
    # Trả về (nhóm, loại): "cưỡng bức" nếu ở thế đó không nước nào tránh được thua ngay; "sai lầm" nếu còn nước an toàn
    # (nước sau đó người thắng không có cột thắng ngay) nhưng người thua không chọn. Chỉ xét MỘT nước cuối của người thua:
    # sai lầm sớm hơn đã đẩy ván vào thế thua cưỡng bức thì không bị phát hiện.
    b = replay(cols[:-2])
    loser_move = int(cols[-2])
    winner = 1 if len(cols) % 2 == 1 else 2
    safe = []
    for m in b.legal_actions():
        after = b.clone()
        after.step(m)
        if not winning_columns(after, winner):
            safe.append(m)
    if safe:
        return "sai lầm", "còn nước an toàn nhưng không chọn"
    w0 = winning_columns(b, winner)
    if len(w0) >= 2:
        return "cưỡng bức", "chiếu đôi"
    if len(w0) == 1 and loser_move == w0[0]:
        return "cưỡng bức", "chặn rồi bị thắng ngay phía trên"
    if len(w0) == 1:
        return "cưỡng bức", "chặn cũng thua"
    return "cưỡng bức", "hết nước an toàn"


def analysis_losses(pairs):
    names = {"ab-d3": "AB-d3", "ab-d4": "AB-d4", "heuristic": "Heuristic", "random": "Random"}
    table = collections.defaultdict(collections.Counter)
    for p in pairs:
        for g in p["games"]:
            if g["winner"] == "draw":
                continue
            seq = [int(m["col"]) for m in p["by_game"][int(g["game"])]]
            loser = names[p["a"] if g["winner"] == p["b"] else p["b"]]
            group, kind = classify_loss(seq)
            table[loser]["tổng"] += 1
            table[loser][group] += 1
            table[loser][kind] += 1
    R["loss_types"] = {a: dict(c) for a, c in table.items()}
    R["loss_note"] = ("phát lại mọi ván có người thắng; chỉ xét nước ngay trước nước thắng cuối cùng của người thắng; "
                      "sai lầm sớm hơn bị bỏ sót")


# ---------- 8. đường cong huấn luyện RL ----------
def read_train(run):
    rows = read_csv(ROOT / "logs" / run / "train_log.csv")
    return dict(ep=[int(r["episode"]) for r in rows], stage=[r["stage"] for r in rows], loss=[float(r["loss"]) if r["loss"] else None for r in rows],
                rnd=[float(r["dev_random_score"]) for r in rows], heu=[float(r["dev_heuristic_score"]) for r in rows],
                el=[float(r["elapsed_s"]) for r in rows], eps=[float(r["epsilon"]) for r in rows])


def analysis_rl():
    names = {"vs_random": ("đấu Random", "#dbeafe"), "vs_heuristic": ("đấu Heuristic", "#fde68a"), "self_pool": ("self-play", "#d1fae5")}
    out = {}
    for run in ("RL-001", "RL-002"):
        d = read_train(run)
        eps, el = d["ep"], d["el"]
        out[run] = dict(evals=len(eps), last_episode=eps[-1], elapsed_s=el[-1], episodes_per_s=round(eps[-1] / el[-1], 2),
                        finished=(ROOT / "logs" / run / "meta.json").exists(),
                        dev_heuristic_max=max(d["heu"]), dev_heuristic_last=d["heu"][-1], dev_random_last=d["rnd"][-1],
                        evals_with_heuristic_1=sum(1 for x in d["heu"] if x >= 0.999),
                        by_stage={})
        for st in names:
            idx = [i for i, s in enumerate(d["stage"]) if s == st]
            if idx:
                h = [d["heu"][i] for i in idx]
                out[run]["by_stage"][st] = dict(evals=len(idx), heuristic_mean=round(statistics.mean(h), 3),
                                                heuristic_min=min(h), heuristic_max=max(h))
        fig, ax = plt.subplots(figsize=(7.4, 3.4))
        start = 0
        for st, (label, color) in names.items():
            idx = [i for i, s in enumerate(d["stage"]) if s == st]
            if idx:
                lo, hi = (eps[idx[0] - 1] if idx[0] > 0 else 0), eps[idx[-1]]
                ax.axvspan(lo, hi, color=color, alpha=0.55, lw=0)
                ax.text((lo + hi) / 2, 1.07, label, ha="center", fontsize=8)
        ax.plot(eps, d["rnd"], color=GRAY, marker=".", lw=1, label="điểm dev vs Random")
        ax.plot(eps, d["heu"], color=BLUE, marker="o", ms=3, lw=1.2, label="điểm dev vs Heuristic")
        ax.set_ylim(-0.03, 1.12)
        ax.set_xlabel("Số ván huấn luyện")
        ax.set_ylabel("Score trên 100 ván dev")
        ax2 = ax.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.plot(eps, d["loss"], color=ORANGE, lw=1, label="loss (trục phải)")
        ax2.set_ylabel("Loss Huber trung bình")
        ln1, lb1 = ax.get_legend_handles_labels()
        ln2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(ln1 + ln2, lb1 + lb2, loc="lower right", frameon=False, fontsize=8)
        ax.set_title(f"{run}: đường cong theo các lần đánh giá dev (mỗi 2.500 ván)", fontsize=10, pad=14)
        fig.tight_layout()
        fig.savefig(OUT / f"fig_rl_curve_{run}.png", dpi=200)
        plt.close(fig)
    R["rl"] = out


# ---------- 9. thí nghiệm chạy thử giữa kỳ ----------
def analysis_pilot():
    rows = read_csv(ROOT / "logs" / "heuristic_vs_random.csv")
    L = [int(r["moves"]) for r in rows]
    first = [int(r["moves"]) for r in rows if r["heuristic_first"] == "1"]
    second = [int(r["moves"]) for r in rows if r["heuristic_first"] == "0"]
    R["pilot"] = dict(games=len(rows), mean=round(statistics.mean(L), 2), median=statistics.median(L), min=min(L), max=max(L),
                      mean_first=round(statistics.mean(first), 2), mean_second=round(statistics.mean(second), 2),
                      distinct_lengths=len(set(L)))
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.hist([first, second], bins=range(7, 30, 2), color=[BLUE, ORANGE], label=["Heuristic đi trước", "Heuristic đi sau"])
    ax.set_xlabel("Số nước đi của một ván")
    ax.set_ylabel("Số ván")
    ax.legend(frameon=False)
    ax.set_title("Thí nghiệm chạy thử: độ dài ván (200 ván)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pilot_lengths.png", dpi=200)
    plt.close(fig)


# ---------- 10. kiểm tra nhanh (KHÔNG phải thí nghiệm chính thức) ----------
def quick_checks():
    positions = {"bàn trống": [], "thế giữa ván (10 nước)": [3, 3, 2, 4, 2, 2, 4, 4, 5, 1]}
    out = {}
    for pname, moves in positions.items():
        rows = []
        for depth in range(2, 7):
            res = {}
            for pruning in (True, False):
                b = replay(moves)
                ag = AlphaBetaAgent(depth=depth, seed=0, pruning=pruning)
                t0 = time.perf_counter()
                action = ag.choose_action(b)
                dt = (time.perf_counter() - t0) * 1000
                res[pruning] = dict(nodes=ag.last_nodes, ms=round(dt, 1), action=action, scores=dict(ag.last_scores))
            rows.append(dict(depth=depth, nodes_on=res[True]["nodes"], nodes_off=res[False]["nodes"], ms_on=res[True]["ms"], ms_off=res[False]["ms"],
                             same_scores=res[True]["scores"] == res[False]["scores"], same_action=res[True]["action"] == res[False]["action"],
                             ratio=round(res[False]["nodes"] / res[True]["nodes"], 2)))
        out[pname] = rows
    R["quick_pruning"] = out
    mc = {}
    b = replay(positions["thế giữa ván (10 nước)"])
    for sims in (200, 1000, 5000):
        speeds = []
        for rep in range(3):
            ag = MCTSAgent(simulations=sims, seed=rep)
            t0 = time.perf_counter()
            ag.choose_action(b)
            speeds.append(sims / (time.perf_counter() - t0))
        mc[str(sims)] = dict(median_sims_per_s=round(statistics.median(speeds)), min=round(min(speeds)), max=round(max(speeds)))
    R["quick_mcts_speed"] = mc


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs()
    assert len(pairs) == 6, "cần đủ 6 cặp của EXP-001"
    analysis_lengths(pairs)
    analysis_first_mover(pairs)
    analysis_ply_cost(pairs)
    analysis_openings(pairs)
    analysis_robustness(pairs)
    analysis_typical(pairs)
    analysis_windows()
    analysis_losses(pairs)
    analysis_rl()
    analysis_pilot()
    quick_checks()
    from rl.network import QNet  # số tham số của mạng Q mặc định (3 conv 64 kênh + FC 128 + FC 7)
    R["qnet_params"] = sum(p.numel() for p in QNet().parameters())
    NUMBERS.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Đã ghi", NUMBERS.name, "và", len(list(OUT.glob("*.png"))), "hình")


if __name__ == "__main__":
    main()
