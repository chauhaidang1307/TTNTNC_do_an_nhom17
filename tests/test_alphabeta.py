import random

import pytest
from agents.alphabeta import CENTER_WEIGHT, WIN, WINDOW_WEIGHT, WINDOWS, AlphaBetaAgent, evaluate
from agents.random_agent import RandomAgent
from engine import board

P1, P2 = board.PLAYER_1, board.PLAYER_2


def play(moves):
    b = board.Board()
    for col in moves:
        b.step(col)
    return b


def test_takes_immediate_win():
    b = play([0, 0, 1, 1, 2, 2])  # người 1 đến lượt, cột 3 thắng ngay
    assert AlphaBetaAgent(depth=2, seed=0).choose_action(b) == 3


def test_blocks_horizontal_win():
    b = play([0, 0, 1, 1, 2, 2, 1])  # người 2 đến lượt, người 1 sắp thắng ở cột 3
    assert AlphaBetaAgent(depth=2, seed=0).choose_action(b) == 3


def test_immediate_win_scores_win_minus_one():
    # thắng ngay ở nước đầu tiên của gốc (ply 1) phải có điểm WIN - 1, kể cả khi tìm sâu hơn
    b = play([0, 0, 1, 1, 2, 2])
    agent = AlphaBetaAgent(depth=4, seed=0)
    assert agent.choose_action(b) == 3
    assert agent.last_scores[3] == WIN - 1


def test_does_not_change_board():
    b = play([3, 3, 2, 4])
    before = [row[:] for row in b.grid], b.to_move, b.moves_made, list(b.history)
    AlphaBetaAgent(depth=3, seed=0).choose_action(b)
    assert ([row[:] for row in b.grid], b.to_move, b.moves_made, list(b.history)) == before


def test_raises_when_game_over():
    b = play([0, 1, 0, 1, 0, 1, 0])  # người 1 thắng theo cột dọc
    assert b.is_terminal()
    with pytest.raises(ValueError):
        AlphaBetaAgent(depth=2, seed=0).choose_action(b)


def test_depth_must_be_positive():
    with pytest.raises(ValueError):
        AlphaBetaAgent(depth=0)


def test_evaluate_empty_board_is_zero():
    assert evaluate(board.Board(), P1) == 0


def test_evaluate_is_antisymmetric():
    b = play([3, 3, 2, 4, 3, 1])
    assert evaluate(b, P1) == -evaluate(b, P2)


def test_evaluate_much_smaller_than_win():
    b = play([3, 3, 3, 3, 3, 3])
    assert abs(evaluate(b, P1)) < WIN // 100


def play_vs_random(agent_seed, random_seed):
    ab = AlphaBetaAgent(depth=2, seed=agent_seed)
    rnd = RandomAgent(seed=random_seed)
    agents = {P1: ab, P2: rnd}
    b = board.Board()
    moves = []
    while not b.is_terminal():
        col = agents[b.to_move].choose_action(b)
        b.step(col)  # step báo lỗi nếu nước không hợp lệ
        moves.append(col)
    return moves


def test_same_seed_same_moves():
    assert play_vs_random(5, 6) == play_vs_random(5, 6)


def test_plays_legal_games_against_random():
    for seed in range(5):
        assert len(play_vs_random(seed, seed + 100)) >= 7


def reference_minimax(b, depth, ply, root, maximizing):
    # minimax viết tường minh bằng max/min (không dùng mẹo đổi dấu) để đối chiếu với negamax
    if b.is_terminal():
        if b.winner is None:
            return 0
        return WIN - ply if b.winner == root else -(WIN - ply)
    if depth == 0:
        return evaluate(b, root)
    values = []
    for col in b.legal_actions():
        b.step(col)
        values.append(reference_minimax(b, depth - 1, ply + 1, root, not maximizing))
        b.undo()
    return max(values) if maximizing else min(values)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_negamax_matches_reference_minimax(depth):
    rng = random.Random(depth)
    checked = 0
    while checked < 8:
        b = board.Board()
        for _ in range(rng.randint(2, 12)):
            if b.is_terminal():
                break
            b.step(rng.choice(b.legal_actions()))
        if b.is_terminal():
            continue
        agent = AlphaBetaAgent(depth=depth, seed=0)
        agent.choose_action(b)
        root = b.to_move
        for col, score in agent.last_scores.items():
            b.step(col)
            expected = reference_minimax(b, depth - 1, 1, root, False)
            b.undo()
            assert score == expected, (depth, col)
        checked += 1


# ---------- Cắt tỉa: bật và tắt công tắc phải cho cùng kết quả ----------

def random_positions(seed, count, min_plies=2, max_plies=14):
    # Sinh `count` thế cờ chưa kết thúc bằng cách đi ngẫu nhiên có seed (tái lập được).
    rng = random.Random(seed)
    positions = []
    while len(positions) < count:
        b = board.Board()
        for _ in range(rng.randint(min_plies, max_plies)):
            if b.is_terminal():
                break
            b.step(rng.choice(b.legal_actions()))
        if not b.is_terminal():
            positions.append(b)
    return positions


def test_pruning_flag_is_stored_and_named():
    on = AlphaBetaAgent(depth=3, pruning=True)
    off = AlphaBetaAgent(depth=3, pruning=False)
    assert on.pruning is True and off.pruning is False
    assert on.name == "AB-d3"
    assert off.name == "AB-d3-nopr"


def test_pruning_defaults_to_on():
    assert AlphaBetaAgent(depth=2).pruning is True


@pytest.mark.parametrize("pruning", [True, False])
def test_takes_immediate_win_in_both_modes(pruning):
    b = play([0, 0, 1, 1, 2, 2])  # người 1 đến lượt, cột 3 thắng ngay
    assert AlphaBetaAgent(depth=3, pruning=pruning, seed=0).choose_action(b) == 3


@pytest.mark.parametrize("pruning", [True, False])
def test_blocks_win_in_both_modes(pruning):
    b = play([0, 0, 1, 1, 2, 2, 1])  # người 2 đến lượt, người 1 sắp thắng ở cột 3
    assert AlphaBetaAgent(depth=3, pruning=pruning, seed=0).choose_action(b) == 3


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_pruning_on_off_same_root_scores_and_action(depth):
    # Tính chất quan trọng nhất: cắt tỉa không được đổi điểm từng nước ở gốc, cũng không đổi nước chọn.
    for b in random_positions(seed=100 + depth, count=15):
        on = AlphaBetaAgent(depth=depth, pruning=True, seed=7)
        off = AlphaBetaAgent(depth=depth, pruning=False, seed=7)
        action_on = on.choose_action(b)
        action_off = off.choose_action(b)
        assert on.last_scores == off.last_scores
        assert action_on == action_off


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_pruning_never_visits_more_nodes(depth):
    for b in random_positions(seed=200 + depth, count=15):
        on = AlphaBetaAgent(depth=depth, pruning=True, seed=0)
        off = AlphaBetaAgent(depth=depth, pruning=False, seed=0)
        on.choose_action(b)
        off.choose_action(b)
        assert on.last_nodes <= off.last_nodes


def test_pruning_visits_strictly_fewer_nodes_somewhere():
    # Nếu cắt tỉa không cắt được nhánh nào thì công tắc vô nghĩa; ở độ sâu 4 phải thấy ít nút hơn.
    fewer = 0
    for b in random_positions(seed=300, count=15):
        on = AlphaBetaAgent(depth=4, pruning=True, seed=0)
        off = AlphaBetaAgent(depth=4, pruning=False, seed=0)
        on.choose_action(b)
        off.choose_action(b)
        if on.last_nodes < off.last_nodes:
            fewer += 1
    assert fewer > 0


def test_depth_two_prunes_nothing_with_full_root_window():
    # Quyết định thiết kế: mỗi nước ở gốc dùng cửa sổ đầy đủ nên ở độ sâu 2 chưa có nhánh nào bị cắt.
    # Nếu sau này đổi cách truyền alpha ở gốc thì test này phải được xem lại có chủ ý.
    for b in random_positions(seed=400, count=10):
        on = AlphaBetaAgent(depth=2, pruning=True, seed=0)
        off = AlphaBetaAgent(depth=2, pruning=False, seed=0)
        on.choose_action(b)
        off.choose_action(b)
        assert on.last_nodes == off.last_nodes


@pytest.mark.parametrize("pruning", [True, False])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_matches_reference_minimax_in_both_modes(depth, pruning):
    # Đối chiếu với minimax viết tường minh bằng max/min; bắt lỗi sai dấu hoặc sai cắt tỉa.
    for b in random_positions(seed=500 + depth, count=6):
        agent = AlphaBetaAgent(depth=depth, pruning=pruning, seed=0)
        agent.choose_action(b)
        root = b.to_move
        for col, score in agent.last_scores.items():
            b.step(col)
            expected = reference_minimax(b, depth - 1, 1, root, False)
            b.undo()
            assert score == expected, (depth, pruning, col)


def play_game_with(pruning, agent_seed, random_seed, depth=3):
    ab = AlphaBetaAgent(depth=depth, pruning=pruning, seed=agent_seed)
    agents = {P1: ab, P2: RandomAgent(seed=random_seed)}
    b = board.Board()
    moves = []
    while not b.is_terminal():
        col = agents[b.to_move].choose_action(b)
        b.step(col)
        moves.append(col)
    return moves


def test_whole_games_identical_with_pruning_on_and_off():
    # Cùng seed, cùng đối thủ Random: hai chế độ phải đi đúng cùng một chuỗi nước.
    for seed in range(5):
        assert play_game_with(True, seed, seed + 50) == play_game_with(False, seed, seed + 50)


# ---------- Lượng giá đếm cửa sổ 4 ô ----------

def test_windows_count_is_69():
    # 24 ngang + 21 dọc + 12 chéo xuôi + 12 chéo ngược
    assert len(WINDOWS) == 69


def test_windows_by_direction():
    counts = {}
    for w in WINDOWS:
        step = (w[1][0] - w[0][0], w[1][1] - w[0][1])
        counts[step] = counts.get(step, 0) + 1
    assert counts == {(0, 1): 24, (1, 0): 21, (1, 1): 12, (1, -1): 12}


def test_windows_are_valid_and_unique():
    seen = set()
    for w in WINDOWS:
        assert len(w) == 4
        for r, c in w:
            assert 0 <= r < board.ROWS and 0 <= c < board.COLS
        key = tuple(w)
        assert key not in seen
        seen.add(key)


def test_cell_bottom_center_is_in_seven_windows():
    # đếm tay: 4 ngang + 1 dọc + 1 chéo xuôi + 1 chéo ngược
    assert sum(1 for w in WINDOWS if (0, 3) in w) == 7


def test_single_piece_at_bottom_center():
    b = play([3])  # một quân của người 1 ở (hàng 0, cột 3)
    expected = 7 * WINDOW_WEIGHT[1] + CENTER_WEIGHT
    assert evaluate(b, P1) == expected
    assert evaluate(b, P2) == -expected


def test_mixed_window_contributes_zero():
    # người 1 ở (0,0), người 2 ở (0,1). Cửa sổ ngang đầu tiên chứa cả hai nên không đóng góp.
    # Cửa sổ chỉ có quân người 1: dọc và chéo xuôi từ (0,0) = 2.
    # Cửa sổ chỉ có quân người 2: ngang bắt đầu ở cột 1, dọc và chéo xuôi từ (0,1) = 3.
    b = play([0, 1])
    assert evaluate(b, P1) == (2 - 3) * WINDOW_WEIGHT[1]
    assert evaluate(b, P2) == (3 - 2) * WINDOW_WEIGHT[1]


def test_three_in_window_scores_higher_than_one():
    one = play([0])
    three = play([0, 6, 1, 5, 2])  # người 1 có ba quân liền ở hàng đáy
    assert evaluate(three, P1) > evaluate(one, P1)


def reference_evaluate(b, player):
    # Cài đặt độc lập, không dùng WINDOWS: duyệt từng ô xuất phát và từng hướng, rồi chấm điểm.
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    points = 0
    for r in range(board.ROWS):
        for c in range(board.COLS):
            for dr, dc in directions:
                cells = [(r + dr * i, c + dc * i) for i in range(4)]
                if not all(0 <= rr < board.ROWS and 0 <= cc < board.COLS for rr, cc in cells):
                    continue
                values = [b.grid[rr][cc] for rr, cc in cells]
                mine = values.count(player)
                theirs = sum(1 for v in values if v not in (board.EMPTY, player))
                if theirs == 0:
                    points += WINDOW_WEIGHT[mine]
                elif mine == 0:
                    points -= WINDOW_WEIGHT[theirs]
    center = board.COLS // 2
    column = [b.grid[r][center] for r in range(board.ROWS)]
    points += CENTER_WEIGHT * (column.count(player) - sum(1 for v in column if v not in (board.EMPTY, player)))
    return points


def test_evaluate_matches_reference_on_random_positions():
    for b in random_positions(seed=600, count=30):
        for player in (P1, P2):
            assert evaluate(b, player) == reference_evaluate(b, player)


def test_evaluate_antisymmetric_on_random_positions():
    for b in random_positions(seed=601, count=30):
        assert evaluate(b, P1) == -evaluate(b, P2)


def test_evaluate_far_below_win_on_random_positions():
    for b in random_positions(seed=602, count=30, max_plies=40):
        assert abs(evaluate(b, P1)) < WIN // 100


def test_window_weights_start_at_zero_and_increase_with_pieces():
    # Ràng buộc thiết kế: cửa sổ trống không đóng góp; càng nhiều quân càng đáng giá.
    # Giá trị cụ thể là giá trị khởi điểm và có thể chỉnh sau bằng ván dev, nhưng hai ràng buộc này phải giữ.
    assert WINDOW_WEIGHT[0] == 0
    assert all(WINDOW_WEIGHT[k] < WINDOW_WEIGHT[k + 1] for k in range(len(WINDOW_WEIGHT) - 1))
    assert len(WINDOW_WEIGHT) == 4  # k = 0..3; k = 4 không xảy ra vì ván kết thúc được xử lý trước


# ---------- Giới hạn thời gian (iterative deepening) ----------

def test_time_limit_is_stored_and_named():
    a = AlphaBetaAgent(depth=10, seed=0, time_limit=0.05)
    assert a.time_limit == 0.05
    assert a.name == "AB-t50ms"
    assert AlphaBetaAgent(depth=10, seed=0, time_limit=0.05, pruning=False).name == "AB-t50ms-nopr"
    assert AlphaBetaAgent(depth=3).time_limit is None


def test_negative_time_limit_rejected():
    with pytest.raises(ValueError):
        AlphaBetaAgent(depth=5, time_limit=-1)


def test_zero_time_still_returns_legal_move_with_depth_one():
    b = play([3, 3, 2, 4])
    a = AlphaBetaAgent(depth=10, seed=0, time_limit=0)
    assert a.choose_action(b) in b.legal_actions()
    assert a.last_depth == 1


def test_time_mode_does_not_change_board():
    b = play([3, 3, 2, 4, 2])
    before = [row[:] for row in b.grid]
    AlphaBetaAgent(depth=10, seed=0, time_limit=0.02).choose_action(b)
    assert b.grid == before


def test_time_mode_respects_depth_cap_and_matches_fixed_depth():
    # Thời gian rất dư, depth là trần: phải ra đúng kết quả của chế độ độ sâu cố định cùng trần.
    for b in random_positions(5, 6):
        timed = AlphaBetaAgent(depth=3, seed=1, time_limit=60)
        fixed = AlphaBetaAgent(depth=3, seed=1)
        assert timed.choose_action(b) == fixed.choose_action(b)
        assert timed.last_depth == 3
        assert timed.last_scores == fixed.last_scores


def test_time_mode_stops_near_the_limit():
    import time
    b = play([3, 3, 2, 4, 2, 2, 4, 4, 5, 1])
    a = AlphaBetaAgent(depth=30, seed=0, time_limit=0.1)
    t0 = time.perf_counter()
    a.choose_action(b)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1 + 0.15  # có dung sai cho kiểm tra đồng hồ mỗi 256 nút và máy chậm
    assert 1 <= a.last_depth < 30


def test_time_mode_takes_immediate_win_and_blocks():
    assert AlphaBetaAgent(depth=10, seed=0, time_limit=0.02).choose_action(play([0, 0, 1, 1, 2, 2])) == 3
    # người 2 đến lượt, người 1 có ba quân ở cột 0..2 hàng dưới -> phải chặn cột 3
    assert AlphaBetaAgent(depth=10, seed=0, time_limit=0.02).choose_action(play([0, 6, 1, 6, 2])) == 3
