import random
import time

from agents.base import Agent
from engine.board import COLS, EMPTY, ROWS, WIN_LENGTH, DIRECTIONS

WIN = 1_000_000
COLUMN_ORDER = [3, 2, 4, 1, 5, 0, 6]  # duyệt cột giữa trước
CENTER_WEIGHT = 3  # giá trị khởi điểm, chưa điều chỉnh qua thực nghiệm
WINDOW_WEIGHT = [0,1,5,20] #Khởi tạo trọng số, chưa qua thực nghiệm

def _build_window():
    # mỗi Windows là danh sách WIN_LENGTH tọa độ (hàng, cột) liên tiếp theo một hướng
    # Chỉ đi xuôi theo mỗi hướng nên không có cửa sổ trùng.
    Windows = []
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in DIRECTIONS:
                end_r = r + dr*(WIN_LENGTH - 1)
                end_c = c + dc*(WIN_LENGTH - 1)
                if 0 <= end_r < ROWS and 0 <= end_c < COLS:
                    Windows.append([(r+dr*i,c+dc*i) for i in range(WIN_LENGTH)])
    return Windows

WINDOWS = _build_window()

def evaluate(board, player): #Hàm lượng giá cho bàn cờ
    points = 0
    for w in WINDOWS:
        mine = 0
        theirs = 0
        for r,c in w:
            cell = board.grid[r][c]
            if cell == player:
                mine += 1
            elif cell != EMPTY:
                theirs += 1
        if theirs == 0: # chỉ có quân của player (hoặc trống hẳn)
            points = points + WINDOW_WEIGHT[mine]
        elif mine == 0: # chỉ có quân đối thủ
            points = points - WINDOW_WEIGHT[theirs]
        # còn lại: cả hai bên cùng có quân -> không đóng góp
    center = COLS // 2
    mine_center = sum(1 for r in range(ROWS) if board.grid[r][center] == player)
    theirs_center = sum(1 for r in range(ROWS) if board.grid[r][center] not in (EMPTY, player))
    points = points + CENTER_WEIGHT * (mine_center - theirs_center)
    return points


class _TimeUp(Exception):
    # Dùng nội bộ: báo hết giờ để thoát khỏi đệ quy ngay lập tức.
    pass


CHECK_EVERY = 256  # kiểm tra đồng hồ mỗi 256 nút (gọi time mỗi nút thì tốn)


class AlphaBetaAgent(Agent):

    INF = 2*WIN

    def __init__(self, depth=4, seed=None, pruning = True, time_limit=None):
        # time_limit=None: tìm đúng độ sâu `depth` (chế độ độ sâu cố định).
        # time_limit=giây: iterative deepening; `depth` là độ sâu tối đa được thử.
        if depth < 1:
            raise ValueError("depth phải >= 1")
        if time_limit is not None and time_limit < 0:
            raise ValueError("time_limit phải >= 0")
        self.depth = depth
        self.seed = seed
        self.pruning = pruning
        self.time_limit = time_limit
        self.rng = random.Random(seed)  # bộ sinh riêng; chỉ gọi đúng một lần mỗi nước đi
        base = f"AB-d{depth}" if time_limit is None else f"AB-t{round(time_limit * 1000)}ms"
        self.name = base if pruning else base + "-nopr"
        self.last_nodes = 0
        self.last_scores = {}
        self.last_depth = 0  # độ sâu hoàn tất của lần chọn nước gần nhất
        self._deadline = None

    def _search_root(self, work, depth):
        # Tìm tới `depth` với cửa sổ đầy đủ ở gốc; trả về {cột: điểm}.
        scores = {}
        for col in self._ordered(work):
            work.step(col)
            scores[col] = -self._negamax(work, depth - 1, 1, -self.INF, self.INF)
            work.undo()
        return scores

    def choose_action(self, board):
        if board.is_terminal():
            raise ValueError("Không còn nước hợp lệ (ván đã kết thúc)")
        work = board.clone()  # Clone một bản sao để làm việc trên đó.
        self.last_nodes = 0
        if self.time_limit is None:
            self._deadline = None
            scores = self._search_root(work, self.depth)
            self.last_depth = self.depth
        else:
            scores = self._iterative_deepening(work, board)
        self.last_scores = scores
        best = max(scores.values())
        top = [c for c in scores if scores[c] == best]
        return self.rng.choice(top)

    def _iterative_deepening(self, work, original):
        # Tìm sâu dần 1, 2, 3, ... đến khi hết giờ hoặc chạm độ sâu tối đa.
        # Lượt tìm bị cắt giữa chừng bị bỏ; dùng kết quả của độ sâu hoàn tất gần nhất.
        # Độ sâu 1 luôn được chạy xong (không kiểm tra giờ) để chắc chắn có nước đi.
        start = time.perf_counter()
        max_depth = min(self.depth, ROWS * COLS - original.moves_made)
        scores = None
        self.last_depth = 0
        for d in range(1, max_depth + 1):
            self._deadline = None if d == 1 else start + self.time_limit
            if d > 1 and time.perf_counter() > self._deadline:
                break  # hết giờ ngay trước khi bắt đầu độ sâu mới
            try:
                scores = self._search_root(work, d)
            except _TimeUp:
                break
            self.last_depth = d
        self._deadline = None
        return scores

    def _ordered(self, board):
        legal = set(board.legal_actions())
        return [c for c in COLUMN_ORDER if c in legal]

    def _negamax(self, board, depth_left, ply, alpha, beta): #giá trị của thế cờ theo góc nhìn của người đang đến lượt tại thế đó.
        self.last_nodes += 1 # Đến số nút đã duyệt, làm chỉ số chi phí.
        if (self._deadline is not None and self.last_nodes % CHECK_EVERY == 0
                and time.perf_counter() > self._deadline):
            raise _TimeUp
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
