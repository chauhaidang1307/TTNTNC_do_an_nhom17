import random

from agents.base import Agent
from engine.board import DIRECTIONS, EMPTY, PLAYER_1, PLAYER_2, COLS, ROWS, WIN_LENGTH

# Luật 6: thứ tự ưu tiên vị trí theo nhóm đối xứng (cột đánh số 0-6, nhóm trước ưu tiên hơn)
POSITION_GROUPS = [[3], [2, 4], [1, 5], [0, 6]]


def other(player):
    return PLAYER_2 if player == PLAYER_1 else PLAYER_1


class HeuristicAgent(Agent):
    # rule-based, ưu tiên cứng theo ghi chú "How Rule-Base Agent work".
    # Rule 1: thắng ngay (nhiều cột thì lấy cột nhỏ nhất)
    # Rule 2: chặn nước thắng ngay của đối thủ (>= 2 cột: chặn ngẫu nhiên một cột, xem Xung đột 4)
    # Rule 4: bộ lọc an toàn, loại nước khiến đối thủ thắng ngay ở lượt sau
    # Rule 5: tạo chuỗi 3 quân có ô hoàn thành chơi được ngay (phụ: số cặp 2 quân)
    # Rule 6: ưu tiên cột giữa; hòa thì chọn ngẫu nhiên theo seed
    # Luật 3 (ô thắng tiềm tàng) KHÔNG có bước chọn riêng: Luật 4 đã mô phỏng nước đi nên tự
    # tránh đặt quân ngay dưới ô thắng của đối thủ. Đây là cách gộp tạm cho mục [CẦN QUYẾT ĐỊNH].
    name = "Heuristic"

    def __init__(self, seed=None):
        self.seed = seed
        self.rng = random.Random(seed)  # bộ sinh riêng để tái lập theo seed

    def choose_action(self, board):
        actions = board.legal_actions()
        if not actions:
            raise ValueError("Không còn nước hợp lệ (ván đã kết thúc)")
        me = board.to_move
        opp = other(me)

        # Luật 1: thắng ngay
        my_wins = self._winning_columns(board, me)
        if my_wins:
            return min(my_wins)

        # Luật 2: chặn đối thủ thắng ngay
        opp_wins = self._winning_columns(board, opp)
        if len(opp_wins) == 1:
            return opp_wins[0]
        if len(opp_wins) >= 2:  # Xung đột 4: không chặn được hết, chặn ngẫu nhiên một cột
            return self.rng.choice(opp_wins)

        # Luật 4: bộ lọc an toàn
        danger = {}
        for col in actions:
            after = board.clone()
            after.step(col)
            danger[col] = 0 if after.is_terminal() else len(self._winning_columns(after, opp))
        safe = [c for c in actions if danger[c] == 0]
        if not safe:  # mọi nước đều nguy hiểm: chọn nước để đối thủ có ít cột thắng nhất
            least = min(danger.values())
            return self.rng.choice([c for c in actions if danger[c] == least])

        # Luật 5: tạo cơ hội tấn công, chỉ xét trong các nước an toàn
        keys = {}
        for col in safe:
            after = board.clone()
            after.step(col)
            keys[col] = self._attack_key(after, me)
        best = max(keys.values())
        candidates = [c for c in safe if keys[c] == best] if best[0] > 0 else safe

        # Luật 6: vị trí chiến lược (đồng thời phá thế hòa theo seed)
        return self._pick_by_position(candidates)

    def _winning_columns(self, board, player):
        # Các cột mà nếu player đặt quân vào thì thắng ngay.
        cols = []
        for col in board.legal_actions():
            sim = board.clone()
            sim.to_move = player
            sim.step(col)
            if sim.winner == player:
                cols.append(col)
        return cols

    def _attack_key(self, after, me):
        # Đánh giá thế sau khi vừa đặt quân, chỉ xét các cửa sổ 4 ô chứa quân vừa đặt
        # và không có quân đối thủ. Trả về (số chuỗi 3 quân có ô trống chơi được ngay, số cặp 2 quân).
        col, row = after.history[-1]
        opp = other(me)
        threes, pairs = 0, 0
        for dr, dc in DIRECTIONS:
            for k in range(WIN_LENGTH):
                r0, c0 = row - k * dr, col - k * dc
                cells = [(r0 + i * dr, c0 + i * dc) for i in range(WIN_LENGTH)]
                if not all(0 <= r < ROWS and 0 <= c < COLS for r, c in cells):
                    continue
                values = [after.grid[r][c] for r, c in cells]
                if opp in values:
                    continue
                mine = values.count(me)
                if mine == 3:
                    er, ec = cells[values.index(EMPTY)]
                    if er == 0 or after.grid[er - 1][ec] != EMPTY:  # ô hoàn thành chơi được ngay
                        threes += 1
                elif mine == 2:
                    pairs += 1
        return (threes, pairs)

    def _pick_by_position(self, candidates):
        for group in POSITION_GROUPS:
            in_group = [c for c in group if c in candidates]
            if in_group:
                return self.rng.choice(in_group)
        return self.rng.choice(candidates)
