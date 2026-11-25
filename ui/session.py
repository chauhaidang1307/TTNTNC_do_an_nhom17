import time
from pathlib import Path

from arena.tournament import make_agent
from engine.board import COLS, DIRECTIONS, EMPTY, PLAYER_1, PLAYER_2, ROWS, WIN_LENGTH, Board
from rl.checkpoint import CHECKPOINT_DIR

HUMAN = "human"
SIDE_NAME = {PLAYER_1: "Đỏ", PLAYER_2: "Vàng"}
SIDE_COLOR = {PLAYER_1: "#e53935", PLAYER_2: "#fdd835"}

# ---- Chọn đối thủ theo CẤP (đúng bảng cấp tác tử của đề) rồi theo ĐỘ MẠNH (Dễ/Vừa/Khó) ----
# Mỗi mức là một tên tác tử của arena. Chi tiết kỹ thuật (độ sâu, số mô phỏng) chỉ hiện ở dòng mô tả, không ở danh sách.
LEVEL_RANDOM = "Cấp 1: Random"
LEVEL_HEURISTIC = "Cấp 2: Heuristic"
LEVEL_ALPHABETA = "Cấp 3: Alpha-Beta"
LEVEL_MCTS = "Cấp 4: MCTS"
LEVEL_RL = "Cấp 6: RL (DQN)"

STRENGTHS = {
    LEVEL_RANDOM: [("Chuẩn", "random")],
    LEVEL_HEURISTIC: [("Chuẩn", "heuristic")],
    LEVEL_ALPHABETA: [("Dễ (độ sâu 2)", "ab-d2"), ("Vừa (độ sâu 4)", "ab-d4"), ("Khó (độ sâu 6)", "ab-d6")],
    LEVEL_MCTS: [("Dễ (200 mô phỏng)", "mcts-s200"), ("Vừa (1000 mô phỏng)", "mcts-s1000"), ("Khó (5000 mô phỏng)", "mcts-s5000")],
}
DEFAULT_STRENGTH = {LEVEL_ALPHABETA: "ab-d4", LEVEL_MCTS: "mcts-s1000"}

LEVEL_INFO = {
    LEVEL_RANDOM: "Đi ngẫu nhiên trong các cột còn chỗ.",
    LEVEL_HEURISTIC: "Thắng ngay nếu có thể, chặn nước thắng của đối thủ, tránh nước nguy hiểm; chưa nhìn xa.",
    LEVEL_ALPHABETA: "Thử trước nhiều nước đi của cả hai bên rồi chọn nước tốt nhất; mức càng cao càng nhìn xa.",
    LEVEL_MCTS: "Chơi thử thật nhiều ván ngẫu nhiên để chọn nước hứa hẹn; mức càng cao càng thử nhiều ván.",
    LEVEL_RL: "Mạng neural tự học bằng cách chơi với chính nó (học tăng cường).",
}


def dqn_choices(root=CHECKPOINT_DIR):
    # Quét checkpoints/<run>/{best,last}.pt. Trả về [(nhãn dễ hiểu, tên tác tử)]; chưa có checkpoint thì rỗng.
    out = []
    root = Path(root)
    if root.exists():
        for run in sorted(p for p in root.iterdir() if p.is_dir()):
            for ckpt, text in (("last", "bản cuối"), ("best", "bản tốt nhất trên ván dev")):
                if (run / f"{ckpt}.pt").exists():
                    out.append((f"{run.name} – {text}", f"dqn_{run.name}_{ckpt}"))
    return out


def level_names(root=CHECKPOINT_DIR):
    # Cấp 6 chỉ có trong danh sách khi đã có checkpoint.
    names = [LEVEL_RANDOM, LEVEL_HEURISTIC, LEVEL_ALPHABETA, LEVEL_MCTS]
    if dqn_choices(root):
        names.append(LEVEL_RL)
    return names


def strength_choices(level, root=CHECKPOINT_DIR):
    if level == LEVEL_RL:
        return dqn_choices(root)
    return STRENGTHS[level]


def default_strength(level, root=CHECKPOINT_DIR):
    choices = strength_choices(level, root)
    wanted = DEFAULT_STRENGTH.get(level)
    values = [v for _, v in choices]
    if wanted in values:
        return wanted
    # Cấp 6: ưu tiên bản "cuối" của lần chạy mới nhất (tên thư mục xếp sau cùng)
    lasts = [v for v in values if v.endswith("_last")]
    return lasts[-1] if lasts else values[0]


def strength_is_selectable(level, root=CHECKPOINT_DIR):
    return len(strength_choices(level, root)) > 1


def describe(level, spec):
    # Một dòng mô tả cấp và mức đang chọn (chi tiết kỹ thuật nằm ở đây).
    text = f"**{level}**: {LEVEL_INFO[level]}"
    if spec.startswith("ab-d"):
        text += f" Mức đang chọn: tìm trước {spec[4:]} nước."
    elif spec.startswith("mcts-s"):
        text += f" Mức đang chọn: {spec[6:]} ván thử mỗi nước."
    elif spec.startswith("dqn_"):
        text += f" Mô hình: {spec[4:].replace('_', ' – ')}."
    return text


def winning_cells(board):
    # Bốn ô thắng (hàng, cột) nếu ván đã có người thắng, ngược lại trả về danh sách rỗng.
    if board.winner is None:
        return []
    p = board.winner
    for r in range(ROWS):
        for c in range(COLS):
            if board.grid[r][c] != p:
                continue
            for dr, dc in DIRECTIONS:
                cells = [(r + dr * i, c + dc * i) for i in range(WIN_LENGTH)]
                if all(0 <= rr < ROWS and 0 <= cc < COLS and board.grid[rr][cc] == p for rr, cc in cells):
                    return cells
    return []


CELL = 64  # kích thước một ô (đơn vị SVG)


def _hole_path(cx, cy, r):
    # Một hình tròn dạng đường vẽ (để đục lỗ khung bằng quy tắc evenodd).
    return f"M{cx - r:.1f},{cy:.1f}a{r:.1f},{r:.1f} 0 1,0 {2 * r:.1f},0a{r:.1f},{r:.1f} 0 1,0 {-2 * r:.1f},0Z"


def render_svg(board, preview_col=None, preview_player=None, animate=False):
    # Vẽ bàn cờ bằng SVG. Hàng 0 là hàng dưới cùng của bàn. Phía trên bàn có một hàng trống để hiện quân xem trước
    # (preview_col, màu của preview_player) và là nơi quân rơi xuống bắt đầu.
    # animate=True: quân vừa đi rơi từ hàng trên xuống ô của nó (hoạt ảnh SVG). Quân nằm PHÍA SAU khung có lỗ
    # tròn, nên khi rơi nó chỉ hiện qua các lỗ, giống bàn cờ thật.
    rows_px, cols_px = ROWS * CELL, COLS * CELL
    top = CELL  # chiều cao hàng xem trước
    foot = 20   # chỗ cho số cột bên dưới
    W, H = cols_px, top + rows_px + foot
    win = set(winning_cells(board))
    last = None
    if board.history:
        c, r = board.history[-1]
        last = (r, c)
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" role="img" '
             f'aria-label="Bàn cờ Connect Four" style="max-width:{W}px;display:block;margin:auto;'
             f'filter:drop-shadow(0 6px 10px rgba(0,0,0,.35))">']

    def center(r, c):
        return c * CELL + CELL / 2, top + (ROWS - 1 - r) * CELL + CELL / 2

    # 1) quân cờ (phía sau khung)
    for r in range(ROWS):
        for c in range(COLS):
            v = board.grid[r][c]
            if v == EMPTY:
                continue
            cx, cy = center(r, c)
            label = "X" if v == PLAYER_1 else "O"  # chữ phụ cho người khó phân biệt màu
            parts.append("<g>")
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="{CELL * 0.43}" fill="{SIDE_COLOR[v]}"/>')
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="{CELL * 0.31}" fill="none" stroke="#000" '
                         f'stroke-opacity="0.18" stroke-width="3"/>')
            parts.append(f'<text x="{cx}" y="{cy + 7}" text-anchor="middle" font-size="20" font-family="sans-serif" '
                         f'font-weight="bold" fill="#000" fill-opacity="0.45">{label}</text>')
            if animate and last == (r, c):
                dy = (ROWS - r) * CELL  # từ hàng xem trước xuống ô này
                dur = 0.3 + 0.08 * (ROWS - 1 - r)
                parts.append(f'<animateTransform attributeName="transform" type="translate" from="0 {-dy}" '
                             f'to="0 0" dur="{dur:.2f}s" fill="freeze" calcMode="spline" keyTimes="0;1" '
                             f'keySplines="0.55 0 1 0.45"/>')
            parts.append("</g>")

    # 2) quân xem trước ở hàng trên
    if preview_col is not None and preview_player is not None and not board.is_terminal():
        pc = max(0, min(COLS - 1, int(preview_col)))
        parts.append(f'<circle class="c4-ghost" cx="{pc * CELL + CELL / 2}" cy="{CELL / 2}" r="{CELL * 0.4}" '
                     f'fill="{SIDE_COLOR[preview_player]}" fill-opacity="0.92" stroke="#000" stroke-opacity="0.35" '
                     f'stroke-width="2" stroke-dasharray="6 4"/>')

    # 3) khung xanh có 42 lỗ tròn (một đường vẽ duy nhất, quy tắc evenodd)
    rr = 14
    frame = (f"M{rr},{top}H{W - rr}A{rr},{rr} 0 0,1 {W},{top + rr}V{top + rows_px - rr}"
             f"A{rr},{rr} 0 0,1 {W - rr},{top + rows_px}H{rr}A{rr},{rr} 0 0,1 0,{top + rows_px - rr}"
             f"V{top + rr}A{rr},{rr} 0 0,1 {rr},{top}Z")
    holes = "".join(_hole_path(*center(r, c), CELL * 0.4) for r in range(ROWS) for c in range(COLS))
    parts.append(f'<path d="{frame}{holes}" fill="#1f4fb0" fill-rule="evenodd" stroke="#123a8a" stroke-width="2"/>')

    # 4) dải sáng của cột đang chọn
    if preview_col is not None and preview_player is not None and not board.is_terminal():
        pc = max(0, min(COLS - 1, int(preview_col)))
        parts.append(f'<rect class="c4-band" x="{pc * CELL}" y="{top}" width="{CELL}" height="{rows_px}" fill="#ffffff" '
                     f'fill-opacity="0.12" pointer-events="none"/>')

    # 5) viền ô thắng, chấm đánh dấu quân vừa đi, số cột
    for (r, c) in sorted(win):
        cx, cy = center(r, c)
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{CELL * 0.43}" fill="none" stroke="#111" stroke-width="5"/>')
    if last is not None and not animate:
        cx, cy = center(*last)
        parts.append(f'<circle cx="{cx}" cy="{cy - CELL * 0.25}" r="4" fill="#111" fill-opacity="0.8"/>')
    for c in range(COLS):
        parts.append(f'<text x="{c * CELL + CELL / 2}" y="{H - 5}" text-anchor="middle" font-size="13" '
                     f'font-family="sans-serif" fill="currentColor" fill-opacity="0.7">{c + 1}</text>')  # hiển thị 1 đến 7
    parts.append("</svg>")
    return "".join(parts)


class GameSession:
    def __init__(self, spec_p1, spec_p2, seed=0):
        self.board = Board()
        self.specs = {PLAYER_1: spec_p1, PLAYER_2: spec_p2}
        self.agents = {
            PLAYER_1: None if spec_p1 == HUMAN else make_agent(spec_p1, seed),
            PLAYER_2: None if spec_p2 == HUMAN else make_agent(spec_p2, seed + 1),
        }
        self.both_human = spec_p1 == HUMAN and spec_p2 == HUMAN  # chế độ người đấu người
        self.hover_col = COLS // 2  # cột đang trỏ/ vừa chọn, dùng để đặt quân bóng khi vẽ lại
        self.moves = []  # mỗi phần tử: dict(player, col, human, name, time_ms, nodes, depth)

    # ----- trạng thái -----
    def is_over(self):
        return self.board.is_terminal()

    def current_is_human(self):
        return not self.is_over() and self.agents[self.board.to_move] is None

    def current_is_ai(self):
        return not self.is_over() and self.agents[self.board.to_move] is not None

    def side_label(self, player):
        spec = self.specs[player]
        if spec == HUMAN:
            who = f"Người chơi {player}" if self.both_human else "Người chơi"
        else:
            who = self.agents[player].name
        return f"{SIDE_NAME[player]} ({who})"

    # ----- nước đi -----
    def play_human(self, col):
        if self.is_over():
            raise ValueError("Ván đã kết thúc.")
        if not self.current_is_human():
            raise ValueError("Chưa đến lượt người chơi.")
        if col not in self.board.legal_actions():
            raise ValueError(f"Cột {col + 1} đã đầy hoặc không hợp lệ.")  # hiển thị 1 đến 7
        player = self.board.to_move
        self.board.step(col)
        self.moves.append(dict(player=player, col=col, human=True,
                               name=f"Người chơi {player}" if self.both_human else "Người chơi",
                               time_ms=None, nodes=None, depth=None))

    def step_ai(self):
        # Cho tác tử đang đến lượt đi một nước. Trả về bản ghi nước đi.
        if self.is_over():
            raise ValueError("Ván đã kết thúc.")
        player = self.board.to_move
        agent = self.agents[player]
        if agent is None:
            raise ValueError("Đang đến lượt người chơi, không phải máy.")
        t0 = time.perf_counter()
        col = agent.choose_action(self.board)
        dt = (time.perf_counter() - t0) * 1000
        self.board.step(col)
        rec = dict(player=player, col=col, human=False, name=agent.name, time_ms=dt,
                   nodes=getattr(agent, "last_nodes", None), depth=getattr(agent, "last_depth", None))
        self.moves.append(rec)
        return rec

    def undo_turn(self):
        # Hoàn tác nước đi gần nhất của người cùng các nước máy đi sau đó. Không có nước người thì không làm gì.
        if not any(m["human"] for m in self.moves):
            return False
        while self.moves:
            m = self.moves.pop()
            self.board.undo()
            if m["human"]:
                break
        return True

    # ----- văn bản hiển thị -----
    def status_text(self):
        b = self.board
        if b.winner is not None:
            return f"**Kết thúc: {self.side_label(b.winner)} thắng.**"
        if b.is_terminal():
            return "**Kết thúc: hòa (bàn cờ đầy).**"
        nxt = b.to_move
        if self.both_human:
            return f"Lượt của {self.side_label(nxt)}. Chọn một cột."
        if self.current_is_human():
            return f"Lượt của bạn: {SIDE_NAME[nxt]}. Chọn một cột."
        return f"Lượt của máy: {self.side_label(nxt)}."

    def last_move_text(self):
        if not self.moves:
            return "Chưa có nước đi."
        m = self.moves[-1]
        base = f"Nước {len(self.moves)}: {SIDE_NAME[m['player']]} ({m['name']}) đi cột {m['col'] + 1}."  # hiển thị 1 đến 7; bên trong vẫn là 0 đến 6
        if m["human"]:
            return base
        extra = [f"{m['time_ms']:.1f} ms"]
        unit = "mô phỏng" if str(m["name"]).startswith("MCTS") else "nút"
        if m["nodes"] is not None:
            extra.append(f"{m['nodes']} {unit}")
        if m["depth"] is not None:
            extra.append(("độ sâu cây " if str(m["name"]).startswith("MCTS") else "độ sâu ") + str(m["depth"]))
        return base + " (" + ", ".join(extra) + ")"

    def render(self, preview_col=None, animate=False):
        # Quân xem trước chỉ hiện khi đang đến lượt người chơi (màu của người đó).
        preview_player = self.board.to_move if self.current_is_human() else None
        if preview_col is None:
            preview_col = self.hover_col
        return render_svg(self.board, preview_col=preview_col, preview_player=preview_player, animate=animate)
