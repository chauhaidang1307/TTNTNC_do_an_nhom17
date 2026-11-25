import math
import random
import time

from agents.base import Agent


class _Node:
    # Một nút của cây. `player` là người VỪA đi nước dẫn tới nút này, và `reward` là tổng điểm
    # tính theo góc nhìn của người đó (thắng 1, hòa 0,5, thua 0: cùng quy ước với score của arena).
    __slots__ = ("parent", "move", "player", "untried", "children", "visits", "reward")

    def __init__(self, parent, move, player, untried):
        self.parent = parent
        self.move = move
        self.player = player
        self.untried = untried  # các nước chưa mở rộng
        self.children = []
        self.visits = 0
        self.reward = 0.0


class MCTSAgent(Agent):
    # MCTS với UCT (UCB1 áp dụng cho cây). Rollout ngẫu nhiên đến hết ván.
    # Mỗi vòng lặp: Selection -> Expansion (thêm MỘT nút con) -> Simulation -> Backpropagation.
    # Nước chọn cuối cùng: nút con của gốc có SỐ lần visited lớn nhất.
    # simulations: số vòng lặp mỗi nước (chế độ số mô phỏng).
    # time_limit : giây mỗi nước; nếu có thì chạy đến hết giờ và BỎ QUA `simulations`
    #              (luôn chạy ít nhất một vòng lặp). Kết quả theo thời gian không tái lập chính xác.
    # c          : hằng số thăm dò của UCB1 (giá trị khởi điểm, chưa điều chỉnh qua thực nghiệm).

    def __init__(self, simulations=1000, seed=None, c=math.sqrt(2), time_limit=None):
        if simulations < 1:
            raise ValueError("simulations phải >= 1")
        if time_limit is not None and time_limit < 0:
            raise ValueError("time_limit phải >= 0")
        self.simulations = simulations
        self.seed = seed
        self.c = c
        self.time_limit = time_limit
        self.rng = random.Random(seed)  # bộ sinh riêng; dùng cho mở rộng, rollout và phá thế hòa
        if time_limit is None:
            self.name = f"MCTS-s{simulations}"
        else:
            self.name = f"MCTS-t{round(time_limit * 1000)}ms"
        self.last_nodes = 0   # số mô phỏng đã chạy ở nước vừa chọn
        self.last_depth = 0   # độ sâu lớn nhất của cây ở nước vừa chọn
        self.last_visits = {}  # {số lần thăm} của các con của gốc
        self.last_values = {}  # {điểm trung bình theo góc nhìn người đi nước đó}

    def choose_action(self, board):
        if board.is_terminal():
            raise ValueError("Không còn nước hợp lệ (ván đã kết thúc)")
        work = board.clone()  # chỉ làm việc trên bản sao; mọi step đều được undo lại sau mỗi vòng lặp
        root = _Node(None, None, None, list(work.legal_actions()))
        deadline = None if self.time_limit is None else time.perf_counter() + self.time_limit
        iterations = 0
        max_depth = 0
        while True:
            if deadline is None:
                if iterations >= self.simulations:
                    break
            elif iterations >= 1 and time.perf_counter() >= deadline:
                break
            max_depth = max(max_depth, self._iterate(root, work))
            iterations += 1

        self.last_nodes = iterations
        self.last_depth = max_depth
        self.last_visits = {ch.move: ch.visits for ch in root.children}
        self.last_values = {ch.move: ch.reward / ch.visits for ch in root.children}
        best = max(ch.visits for ch in root.children)
        top = [ch.move for ch in root.children if ch.visits == best]
        return self.rng.choice(top)  # phá thế hòa theo seed; gọi đúng một lần mỗi nước

    def _iterate(self, root, work):
        # Một vòng lặp MCTS. Trả về độ sâu của nút đã đạt tới. `work` được trả về đúng trạng thái ban đầu.
        steps = 0
        node = root

        # Selection: đi xuống khi nút đã mở rộng hết và còn con
        while not node.untried and node.children:
            node = self._select_child(node)
            work.step(node.move)
            steps += 1

        # Expansion: nếu chưa hết ván và còn nước chưa thử, thêm MỘT nút con
        if node.untried and not work.is_terminal():
            move = node.untried.pop(self.rng.randrange(len(node.untried)))
            player = work.to_move
            work.step(move)
            steps += 1
            child = _Node(node, move, player, list(work.legal_actions()))
            node.children.append(child)
            node = child

        depth = steps

        # Simulation: chơi ngẫu nhiên đến hết ván
        rollout_steps = 0
        while not work.is_terminal():
            work.step(self.rng.choice(work.legal_actions()))
            rollout_steps += 1
        winner = work.winner

        # Backpropagation: từ nút mới lên gốc; mỗi nút nhận điểm theo góc nhìn của người đã đi vào nút đó
        while node is not None:
            node.visits += 1
            if node.player is not None:  # gốc không có người "vừa đi"
                node.reward += 0.5 if winner is None else (1.0 if winner == node.player else 0.0)
            node = node.parent

        for _ in range(steps + rollout_steps):  # trả bàn cờ về trạng thái gốc
            work.undo()
        return depth

    def _select_child(self, node):
        # UCB1: trung bình + c * sqrt(ln(N_cha) / n_con). Điểm trung bình của con đã ở góc nhìn của
        # người đi nước vào con, chính là người đang chọn tại `node`, nên không cần đổi dấu.
        log_n = math.log(node.visits)
        best_score = -1.0
        best = None
        for ch in node.children:
            s = ch.reward / ch.visits + self.c * math.sqrt(log_n / ch.visits)
            if s > best_score:
                best_score = s
                best = ch
        return best
