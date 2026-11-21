import random

from agents.base import Agent
from engine.board import COLS, EMPTY, ROWS

WIN = 1_000_000
COLUMN_ORDER = [3, 2, 4, 1, 5, 0, 6]  # duyệt cột giữa trước
CENTER_WEIGHT = 3  # giá trị khởi điểm, chưa chỉnh bằng ván dev


def evaluate(board, player):
    # Lượng giá thế cờ theo góc nhìn của `player` (bản đầu: chỉ đếm quân ở cột giữa)
    # Đối xứng: evaluate(b, người 1) == -evaluate(b, người 2)
    center = COLS // 2
    mine = sum(1 for r in range(ROWS) if board.grid[r][center] == player)
    theirs = sum(1 for r in range(ROWS) if board.grid[r][center] not in (EMPTY, player))
    return CENTER_WEIGHT * (mine - theirs)


class AlphaBetaAgent(Agent):
    
    INF = 2*WIN

    def __init__(self, depth=4, seed=None, pruning = True):
        if depth < 1:
            raise ValueError("depth phải >= 1")
        self.depth = depth
        self.seed = seed
        self.pruning = pruning
        self.rng = random.Random(seed)  # bộ sinh riêng; chỉ gọi đúng một lần mỗi nước đi
        if self.pruning:
            self.name = f"AB-d{depth}"
        else:
            self.name = f"AB-d{depth}-nopr"
        self.last_nodes = 0
        self.last_scores = {}

    def choose_action(self, board):
        if board.is_terminal():
            raise ValueError("Không còn nước hợp lệ (ván đã kết thúc)")
        work = board.clone()  # Clone một bản sao để làm việc trên đó.
        self.last_nodes = 0
        scores = {}
        for col in self._ordered(work):
            work.step(col)
            scores[col] = -self._negamax(work, self.depth - 1, 1, -self.INF, self.INF)
            work.undo()
        self.last_scores = scores
        best = max(scores.values())
        top = [c for c in scores if scores[c] == best]
        return self.rng.choice(top)

    def _ordered(self, board):
        legal = set(board.legal_actions())
        return [c for c in COLUMN_ORDER if c in legal]

    def _negamax(self, board, depth_left, ply, alpha, beta): #giá trị của thế cờ theo góc nhìn của người đang đến lượt tại thế đó.
        self.last_nodes += 1 # Đến số nút đã duyệt, làm chỉ số chi phí.
        if board.is_terminal(): # Nếu như bàn cờ đã kết thúc.
            if board.winner is None: # Mà không có người chiến thắng.
                return 0 #Kết quả hòa
            return -(WIN - ply)  # Nếu người vừa đi là người thắng, thì người đến lượt là người thua nên trả về giá trị âm.
        if depth_left == 0: #Không còn độ sâu cho phép
            return evaluate(board, board.to_move) # ước lượng thế cờ theo góc nhìn của người đến lượt.
        best = -self.INF # khởi tạo giá trị đầu cho `best`, mọi giá trị gặp là khoảng [-WIN,WIN] là WIN-ply, vì vậy -WIN*2 dùng làm điểm khởi tạo để nằm ngoài phạm vi có thể trùng vào kết quả.
        for col in self._ordered(board): # duyệt từng nước cờ hợp lệ
            board.step(col) #thử đi step
            if self.pruning:
                value = -self._negamax(board, depth_left - 1, ply + 1, -beta, -alpha) # gọi lại negamax cho đến ván kết thúc hoặc depth = 0, truyền vào tham số alpha beta khi bật pruning
            else:
                value = -self._negamax(board, depth_left - 1, ply + 1,-self.INF,+self.INF) # gọi lại negamax cho đến ván kết thúc hoặc depth = 0
            board.undo() # hoàn tác lại bàn cờ 
            if value > best: #nếu giá trị lớn hơn best
                best = value #gán best = giá trị.
            if self.pruning:
                alpha = max(alpha,best)
                if alpha >= beta: #Phần còn lại không còn ảnh hưởng đến quyết định.
                    break
        return best #trả về giá trị thế cờ
