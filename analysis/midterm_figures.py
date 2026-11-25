from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch

from agents.heuristic import HeuristicAgent
from agents.random_agent import RandomAgent
from engine.board import Board, COLS, PLAYER_1, PLAYER_2, ROWS

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"

THEMES = {
    "dark": dict(bg="#1C2C3B", fg="#FFFFFF", muted="#96BFCB", a="#F88196", b="#96BFCB", panel="#44466A", empty="#2A3E52"),
    "light": dict(bg="#FFFFFF", fg="#1C2C3B", muted="#5A6B7B", a="#D9506A", b="#3E8FA3", panel="#E6E8F2", empty="#E9EEF3"),
}
plt.rcParams["font.family"] = "DejaVu Sans"


def style(ax, t):
    ax.set_facecolor(t["bg"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["muted"])
    ax.tick_params(colors=t["fg"])
    ax.yaxis.label.set_color(t["fg"])
    ax.xaxis.label.set_color(t["fg"])
    ax.title.set_color(t["fg"])


def replay(row):
    # Chạy lại ván Heuristic (người 1) vs Random (người 2) theo seed; kiểm tra khớp log trước khi vẽ.
    agents = {PLAYER_1: HeuristicAgent(seed=int(row.seed_1)), PLAYER_2: RandomAgent(seed=int(row.seed_2))}
    board = Board()
    while not board.is_terminal():
        board.step(agents[board.to_move].choose_action(board))
    assert board.winner == row.winner and board.moves_made == row.so_nuoc, "ván chạy lại không khớp log"
    return board


def draw_board(ax, board, t, label_fs=11):
    ax.set_facecolor(t["bg"])
    ax.set_xlim(-0.1, COLS + 0.1)
    ax.set_ylim(-0.9, ROWS + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), COLS, ROWS, boxstyle="round,pad=0.02,rounding_size=0.15", fc=t["panel"], ec="none"))
    colors = {0: t["empty"], PLAYER_1: t["a"], PLAYER_2: t["b"]}
    for r in range(ROWS):
        for c in range(COLS):
            ax.add_patch(Circle((c + 0.5, r + 0.5), 0.38, fc=colors[board.grid[r][c]], ec="none"))
    for c in range(COLS):
        ax.text(c + 0.5, -0.45, str(c), ha="center", va="center", color=t["fg"], fontsize=label_fs)


def saved_boards(name, t):
    saved = pd.read_csv(ROOT / "logs" / "saved_seeds.csv")
    rows = saved[(saved.agent_1 == "Heuristic") & (saved.agent_2 == "Random")]
    # hình đơn cho ván đầu tiên
    first = rows.iloc[0]
    fig, ax = plt.subplots(figsize=(4.2, 3.9), dpi=200)
    fig.patch.set_facecolor(t["bg"])
    draw_board(ax, replay(first), t, 13)
    fig.tight_layout()
    fig.savefig(OUT / name / "board_example.png", facecolor=t["bg"])
    plt.close(fig)
    # hình ghép cả ba ván đã lưu
    fig, axes = plt.subplots(1, len(rows), figsize=(3.6 * len(rows), 3.5), dpi=200)
    fig.patch.set_facecolor(t["bg"])
    for ax, (_, row) in zip(axes, rows.iterrows()):
        draw_board(ax, replay(row), t, 10)
        ax.set_title(f"seed {int(row.seed_1)} / {int(row.seed_2)}: {int(row.so_nuoc)} nước", color=t["fg"], fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / name / "board_saved3.png", facecolor=t["bg"])
    plt.close(fig)
    return rows


def results_chart(df, name, t):
    groups = [("Heuristic đi trước", df[df.heuristic_first == 1]), ("Heuristic đi sau", df[df.heuristic_first == 0]), ("Tổng", df)]
    fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=220)
    fig.patch.set_facecolor(t["bg"])
    style(ax, t)
    ax.tick_params(labelsize=10)
    for i, (label, sub) in enumerate(groups):
        win = (sub.winner == "Heuristic").sum()
        draw = (sub.winner == "draw").sum()
        loss = (sub.winner == "Random").sum()
        left = 0
        for v, col in ((win, t["a"]), (draw, t["muted"]), (loss, t["b"])):
            if v:
                ax.barh(i, v, left=left, color=col, height=0.55)
                ax.text(left + v / 2, i, str(v), ha="center", va="center", color="#1C2C3B", fontsize=10, fontweight="bold")
            left += v
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[0] for g in groups])
    ax.invert_yaxis()
    ax.set_xlabel("Số ván", fontsize=10)
    ax.set_title("Thắng (hồng), hòa (xám), thua (xanh)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / name / "results_h_vs_r.png", facecolor=t["bg"])
    plt.close(fig)


def moves_hist(df, name, t):
    fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=220)
    fig.patch.set_facecolor(t["bg"])
    style(ax, t)
    ax.tick_params(labelsize=10)
    ax.hist(df.moves, bins=range(int(df.moves.min()), int(df.moves.max()) + 2), color=t["a"], edgecolor=t["bg"])
    ax.set_xlabel("Số nước của ván (cả hai bên)", fontsize=10)
    ax.set_ylabel("Số ván", fontsize=10)
    ax.set_title("Phân bố độ dài ván", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / name / "moves_hist.png", facecolor=t["bg"])
    plt.close(fig)


def architecture(name, t):
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=220)
    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["bg"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, fs=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=t["panel"], ec=t["a"], lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=t["fg"], fontsize=fs)

    arrow = dict(arrowstyle="->", color=t["muted"], lw=1.2)
    box(0.4, 3.75, 4.2, 1.0, "ui/cli.py\nDemo máy-máy: Heuristic vs Random")
    box(5.2, 3.75, 4.4, 1.0, "arena/heuristic_vs_random.py\nchạy thử 200 ván, ghi log thô")
    ax.text(5.0, 3.4, "agents/: Agent.choose_action(board)", ha="center", color=t["muted"], fontsize=9)
    box(1.6, 2.2, 2.8, 0.9, "Random\n(agents/random_agent.py)")
    box(5.6, 2.2, 2.8, 0.9, "Heuristic, 6 luật\n(agents/heuristic.py)")
    box(0.4, 0.45, 5.4, 1.15, "engine/board.py: Board\nreset, legal_actions, step, undo,\nclone, is_terminal, result")
    box(6.2, 0.45, 3.4, 1.15, "tests/ (pytest, 44 test)\nlogs/ (log thô, seed)")
    ax.annotate("", xy=(2.5, 1.62), xytext=(2.5, 2.15), arrowprops=arrow)
    ax.annotate("", xy=(4.6, 1.62), xytext=(6.2, 2.15), arrowprops=arrow)
    ax.annotate("", xy=(2.5, 3.15), xytext=(2.5, 3.8), arrowprops=arrow)
    ax.annotate("", xy=(7.0, 3.15), xytext=(7.0, 3.8), arrowprops=arrow)
    fig.tight_layout()
    fig.savefig(OUT / name / "architecture.png", facecolor=t["bg"])
    plt.close(fig)


def main():
    df = pd.read_csv(ROOT / "logs" / "heuristic_vs_random.csv")
    for name, t in THEMES.items():
        (OUT / name).mkdir(parents=True, exist_ok=True)
        results_chart(df, name, t)
        moves_hist(df, name, t)
        rows = saved_boards(name, t)
        architecture(name, t)
    print("Wrote", OUT, "| saved games replayed:", len(rows))


if __name__ == "__main__":
    main()
