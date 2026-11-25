import collections
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "figures" / "report"
EXP = ROOT / "logs" / "EXP-001"

BLUE, ORANGE, GREEN, PURPLE, GRAY = "#2563eb", "#d97706", "#059669", "#7c3aed", "#6b7280"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load_pairs():
    pairs = []
    for f in sorted(EXP.glob("*/*_meta.json")):
        meta = json.loads(f.read_text(encoding="utf-8"))
        moves = f.with_name(f.name.replace("_meta.json", "_moves.csv"))
        seqs = collections.defaultdict(list)
        with open(moves, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                seqs[int(r["game"])].append(r["col"])
        meta["distinct"] = len({tuple(v) for v in seqs.values()})
        meta["n_games"] = len(seqs)
        pairs.append(meta)
    return pairs


def fig_scores(pairs):
    labels = [f"{p['agent_a']} vs {p['agent_b']}" for p in pairs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    y = range(len(pairs))
    sc = [p["summary"]["total"]["score"] for p in pairs]
    lo = [p["summary"]["total"]["score"] - p["summary"]["total"]["wilson_low"] for p in pairs]
    hi = [p["summary"]["total"]["wilson_high"] - p["summary"]["total"]["score"] for p in pairs]
    ax1.errorbar(sc, list(y), xerr=[lo, hi], fmt="o", color=BLUE, ecolor=BLUE, capsize=3)
    ax1.axvline(0.5, color=GRAY, ls="--", lw=1)
    ax1.set_yticks(list(y))
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlim(-0.02, 1.0)
    ax1.set_xlabel("Score của agent đứng trước (điểm và khoảng Wilson 95%)")
    ax1.set_title("(a) Score tổng, 100 ván mỗi cặp", fontsize=10)
    h = 0.38
    first = [p["summary"]["a_first"]["score"] for p in pairs]
    second = [p["summary"]["a_second"]["score"] for p in pairs]
    ax2.barh([i - h / 2 for i in y], first, h, color=BLUE, label="đi trước")
    ax2.barh([i + h / 2 for i in y], second, h, color=ORANGE, label="đi sau")
    ax2.axvline(0.5, color=GRAY, ls="--", lw=1)
    ax2.set_yticks(list(y))
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel("Score của agent đứng trước, tách theo lượt đi")
    ax2.set_title("(b) Theo lượt đi", fontsize=10)
    for i, (f, sd) in enumerate(zip(first, second)):
        ax2.text(f + 0.012, i - h / 2, f"{f:.2f}", va="center", fontsize=7.5, color=BLUE)
        ax2.text(sd + 0.012, i + h / 2, f"{sd:.2f}", va="center", fontsize=7.5, color=ORANGE)
    ax2.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_exp001_scores.png", dpi=200)
    plt.close(fig)


def fig_cost(pairs):
    order = ["Random", "Heuristic", "AB-d3", "AB-d4"]
    pts = collections.defaultdict(list)
    for p in pairs:
        for name, t in p["timing"].items():
            pts[name].append((t["mean_ms"], t["p95_ms"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, name in enumerate(order):
        vals = pts[name]
        xs = [i + (k - (len(vals) - 1) / 2) * 0.09 for k in range(len(vals))]
        ax.scatter(xs, [v[0] for v in vals], color=BLUE, s=26, label="trung bình" if i == 0 else None, zorder=3)
        ax.scatter(xs, [v[1] for v in vals], color=ORANGE, marker="x", s=30, label="p95" if i == 0 else None, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("Thời gian mỗi nước (ms, thang log)")
    ax.set_title("Thời gian mỗi nước; mỗi chấm là một cặp đối thủ khác nhau", fontsize=10)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#e5e7eb", zorder=0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_exp001_cost.png", dpi=200)
    plt.close(fig)


def fig_distinct(pairs):
    labels = [f"{p['agent_a']} vs {p['agent_b']}" for p in pairs]
    vals = [p["distinct"] for p in pairs]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    bars = ax.barh(labels, vals, color=BLUE)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Số chuỗi nước đi khác nhau trong 100 ván")
    for b, v in zip(bars, vals):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=9)
    ax.set_title("Mức trùng lặp giữa các ván (100 nghĩa là không ván nào trùng)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_exp001_distinct.png", dpi=200)
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, ec, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, color=GRAY, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle=style, color=color, lw=1.2))


def fig_notation():
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.set_xlim(-0.9, 7.2)
    ax.set_ylim(-0.9, 6.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 7, 6, boxstyle="round,pad=0,rounding_size=0.15", fc="#1f4fb0", ec="#123a8a"))
    for r in range(6):
        for c in range(7):
            ax.add_patch(Circle((c + 0.5, r + 0.5), 0.38, fc="#f2f4f8", ec="#123a8a", lw=0.6))
    for c in range(7):
        ax.text(c + 0.5, -0.45, str(c + 1), ha="center", fontsize=9)
    for r in range(6):
        ax.text(-0.45, r + 0.5, str(r + 1), ha="center", va="center", fontsize=9)
    ax.text(3.5, -0.85, "cột (1 đến 7, trái sang phải)", ha="center", fontsize=8, color=GRAY)
    ax.text(-0.9, 3.0, "hàng (1 là dưới cùng)", rotation=90, va="center", ha="center", fontsize=8, color=GRAY)
    lines = [("ngang", [(0, 0), (0, 1), (0, 2), (0, 3)], BLUE), ("dọc", [(2, 6), (3, 6), (4, 6), (5, 6)], GREEN),
             ("chéo lên", [(1, 3), (2, 4), (3, 5), (4, 6)], ORANGE), ("chéo xuống", [(5, 0), (4, 1), (3, 2), (2, 3)], PURPLE)]
    for name, cells, color in lines:
        for (r, c) in cells:
            ax.add_patch(Circle((c + 0.5, r + 0.5), 0.40, fc="none", ec=color, lw=2.6))
        ax.plot([], [], color=color, lw=3, label=name)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=4, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_notation.png", dpi=200)
    plt.close(fig)


def _node(ax, x, y, fc="white", ec=GRAY, r=0.06, lw=1.4):
    from matplotlib.patches import Circle
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=3))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.22, 0.82, 0.56, 0.16, "engine\nBoard: luật chơi, step, undo, clone\nencoding: tensor 2x6x7", "#dbeafe", BLUE, fs=8.5)
    ax.text(0.02, 0.755, "agents: cùng giao diện choose_action(board), chỉ đọc bàn cờ", fontsize=8, color=GRAY)
    names = ["Random\ncấp 1", "Heuristic\ncấp 2", "Alpha-Beta\ncấp 3", "MCTS-UCT\ncấp 4", "DQN\ncấp 6"]
    colors = ["#e5e7eb", "#e5e7eb", "#d1fae5", "#d1fae5", "#fef3c7"]
    edges = [GRAY, GRAY, GREEN, GREEN, ORANGE]
    xs = [0.02, 0.215, 0.41, 0.605, 0.80]
    for n, c, e, x in zip(names, colors, edges, xs):
        box(ax, x, 0.58, 0.18, 0.14, n, c, e, fs=8.5)
    arrow(ax, 0.50, 0.82, 0.50, 0.72, BLUE)
    box(ax, 0.02, 0.28, 0.36, 0.22, "arena\n(độc lập với giao diện)\nmatch, tournament, metrics\nround-robin, seed dev/test", "#ede9fe", PURPLE, fs=8.3)
    box(ax, 0.40, 0.28, 0.26, 0.22, "ui: Gradio\n3 chế độ chơi\nsession.py, gui.py", "#fce7f3", "#be185d", fs=8.3)
    box(ax, 0.68, 0.28, 0.30, 0.22, "rl: network, replay,\ntrainer (Double DQN),\ncheckpoint", "#fef3c7", ORANGE, fs=8.3)
    arrow(ax, 0.20, 0.58, 0.20, 0.50, PURPLE)
    arrow(ax, 0.53, 0.58, 0.53, 0.50, "#be185d")
    arrow(ax, 0.89, 0.50, 0.89, 0.58, ORANGE)
    box(ax, 0.02, 0.04, 0.36, 0.17, "logs/<EXP>/<A>_vs_<B>/\ngames, moves, meta", "#f3f4f6", GRAY, fs=8.3)
    arrow(ax, 0.20, 0.28, 0.20, 0.21, PURPLE)
    box(ax, 0.40, 0.04, 0.58, 0.17, "tests/ (226 test)  |  configs/  |  analysis/\n(hình và bảng sinh lại từ log)", "#f3f4f6", GRAY, fs=8.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_architecture.png", dpi=220)
    plt.close(fig)


def fig_mcts_cycle():
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.7))
    titles = ["1. Selection", "2. Expansion", "3. Simulation", "4. Backprop."]
    base = {"root": (0.5, 0.88), "a": (0.25, 0.62), "b": (0.75, 0.62), "c": (0.12, 0.36), "d": (0.38, 0.36), "e": (0.62, 0.36)}
    edges = [("root", "a"), ("root", "b"), ("a", "c"), ("a", "d"), ("b", "e")]
    for i, ax in enumerate(axes):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(titles[i], fontsize=9)
        for u, v in edges:
            ax.plot([base[u][0], base[v][0]], [base[u][1], base[v][1]], color=GRAY, lw=1, zorder=1)
        for k, (x, y) in base.items():
            _node(ax, x, y, fc="#f3f4f6")
        for u, v in (("root", "a"), ("a", "d")):
            ax.plot([base[u][0], base[v][0]], [base[u][1], base[v][1]], color=BLUE, lw=2.4, zorder=2)
        for k in ("root", "a", "d"):
            _node(ax, *base[k], fc="#dbeafe", ec=BLUE, lw=1.8)
        if i >= 1:
            ax.plot([0.38, 0.38], [0.36, 0.12], color=ORANGE, lw=1.8, ls="--", zorder=2)
            _node(ax, 0.38, 0.12, fc="#fef3c7", ec=ORANGE, lw=1.8)
        if i == 2:
            ax.annotate("", xy=(0.82, 0.05), xytext=(0.45, 0.12), arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.6, connectionstyle="arc3,rad=-0.4"))
            ax.text(0.72, 0.19, "chơi ngẫu\nnhiên", fontsize=7.5, color=PURPLE, ha="center")
            ax.text(0.90, 0.04, "T/H/B", fontsize=7.5, color=PURPLE, weight="bold", va="center")
        if i == 3:
            for (x0, y0, x1, y1) in ((0.38, 0.17, 0.38, 0.30), (0.36, 0.41, 0.27, 0.56), (0.27, 0.67, 0.47, 0.82)):
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.8))
            ax.text(0.76, 0.14, "cập nhật\nn và w", fontsize=7.5, color=GREEN, ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "fig_mcts_cycle.png", dpi=220)
    plt.close(fig)


def fig_qnet():
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    row1 = [("Đầu vào\n2 x 6 x 7\n(người đi, đối thủ)", "#dbeafe", BLUE), ("Conv 3x3\n64 kênh + ReLU", "#d1fae5", GREEN),
            ("Conv 3x3\n64 kênh + ReLU", "#d1fae5", GREEN), ("Conv 3x3\n64 kênh + ReLU", "#d1fae5", GREEN)]
    row2 = [("Flatten\n64 x 6 x 7 = 2.688", "#f3f4f6", GRAY), ("FC 128\n+ ReLU", "#fef3c7", ORANGE),
            ("FC 7\n(Q của 7 cột)", "#fef3c7", ORANGE), ("Mặt nạ\nnước hợp lệ\n+ argmax", "#ede9fe", PURPLE)]
    w, gap = 0.20, 0.0533
    for rowi, (row, y) in enumerate(((row1, 0.58), (row2, 0.14))):
        for i, (t, fc, ec) in enumerate(row):
            x = 0.02 + i * (w + gap)
            ax.add_patch(FancyBboxPatch((x, y), w, 0.28, boxstyle="round,pad=0.005,rounding_size=0.01", fc=fc, ec=ec, lw=1.4))
            ax.text(x + w / 2, y + 0.14, t, ha="center", va="center", fontsize=8.5)
            if i < len(row) - 1:
                ax.annotate("", xy=(x + w + gap - 0.003, y + 0.14), xytext=(x + w + 0.003, y + 0.14), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.2))
    x_last = 0.02 + 3 * (w + gap) + w / 2   # giữa hộp cuối hàng 1
    x_first = 0.02 + w / 2                  # giữa hộp đầu hàng 2
    ax.plot([x_last, x_last, x_first], [0.58, 0.50, 0.50], color=GRAY, lw=1.2)
    ax.annotate("", xy=(x_first, 0.42), xytext=(x_first, 0.50), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.2))
    ax.text(0.5, 0.02, "Padding 1 giữ nguyên kích thước 6 x 7 qua ba lớp tích chập; 64 x 6 x 7 = 2.688.", ha="center", fontsize=8, color=GRAY)
    fig.tight_layout()
    fig.savefig(OUT / "fig_qnet.png", dpi=220)
    plt.close(fig)


def fig_rl_pipeline():
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    W, H = 0.29, 0.20
    boxes = [(0.02, 0.70, "1. Chơi một ván\n(học vs đối thủ),\nghi MỌI nước", "#dbeafe", BLUE),
             (0.355, 0.70, "2. Bộ nhớ kinh nghiệm\n200.000 bước\n(s, a, r, s', done, mặt nạ)", "#f3f4f6", GRAY),
             (0.69, 0.70, "3. Lấy mẫu 256 bước\n+ lật gương ngẫu\nnhiên", "#fef3c7", ORANGE),
             (0.69, 0.40, "4. Mục tiêu Double DQN\nkiểu negamax:\ny = r - g Q_đích(s', a*)", "#d1fae5", GREEN),
             (0.355, 0.40, "5. Huber loss, Adam,\ncắt gradient 10\n(4 bước mỗi ván)", "#ede9fe", PURPLE),
             (0.02, 0.40, "6. Mỗi 2.000 bước:\nsao chép sang\nmạng đích", "#f3f4f6", GRAY),
             (0.02, 0.08, "7. Mỗi 2.500 ván: đánh giá\n100 ván vs Random, 100 vs\nHeuristic (dev); lưu checkpoint", "#fce7f3", "#be185d")]
    for x, y, t, fc, ec in boxes:
        ax.add_patch(FancyBboxPatch((x, y), W if x < 0.5 or y != 0.08 else W, H, boxstyle="round,pad=0.005,rounding_size=0.01", fc=fc, ec=ec, lw=1.4))
        ax.text(x + W / 2, y + H / 2, t, ha="center", va="center", fontsize=8)

    def arr(x1, y1, x2, y2, color=GRAY, style="-|>", rad=0.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle=style, color=color, lw=1.3, connectionstyle=f"arc3,rad={rad}"))
    arr(0.31, 0.80, 0.355, 0.80)
    arr(0.645, 0.80, 0.69, 0.80)
    arr(0.835, 0.70, 0.835, 0.60)
    arr(0.69, 0.50, 0.645, 0.50)
    arr(0.355, 0.50, 0.31, 0.50)
    arr(0.165, 0.40, 0.165, 0.28)
    ax.annotate("", xy=(0.02, 0.80), xytext=(0.02, 0.18), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4, connectionstyle="arc3,rad=-0.28"))
    ax.text(-0.075, 0.49, "mạng mới sinh ván mới", fontsize=7.5, color=BLUE, rotation=90, va="center", ha="center")
    ax.set_xlim(-0.10, 1)
    ax.text(0.5, 0.985, "Đối thủ theo giai đoạn: 20.000 ván đấu Random; 40.000 ván đấu Heuristic (epsilon 0,1);\n140.000 ván self-play với kho bản cũ", ha="center", va="top", fontsize=7.8, color=GRAY)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rl_pipeline.png", dpi=220)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs()
    fig_scores(pairs)
    fig_cost(pairs)
    fig_distinct(pairs)
    fig_architecture()
    fig_notation()
    fig_mcts_cycle()
    fig_qnet()
    fig_rl_pipeline()
    print("Đã ghi:", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
