import pytest
from engine import board
from agents.heuristic import HeuristicAgent
from agents.random_agent import RandomAgent

P1, P2 = board.PLAYER_1, board.PLAYER_2


def play(moves):
    b = board.Board()
    for col in moves:
        b.step(col)
    return b


def test_first_move_is_center():
    for seed in range(10):
        assert HeuristicAgent(seed=seed).choose_action(board.Board()) == 3


def test_takes_immediate_win():
    b = play([0, 0, 1, 1, 2, 2])  # người 1 đến lượt, cột 3 thắng ngay
    assert b.to_move == P1
    assert HeuristicAgent(seed=0).choose_action(b) == 3


def test_blocks_horizontal_win():
    b = play([0, 0, 1, 1, 2, 2, 1])  # người 2 đến lượt, người 1 sắp thắng ở cột 3
    assert b.to_move == P2
    assert HeuristicAgent(seed=0).choose_action(b) == 3


def test_blocks_vertical_win():
    b = play([0, 1, 0, 1, 0])  # người 1 có 3 quân xếp chồng ở cột 0
    assert b.to_move == P2
    assert HeuristicAgent(seed=0).choose_action(b) == 0


def test_win_beats_block():
    # Xung đột 1: người 1 thắng được ở cột 3, người 2 thắng được ở cột 6 -> phải chọn thắng.
    b = board.Board()
    for c in (0, 1, 2):
        b.grid[0][c] = P1
    for r in (0, 1, 2):
        b.grid[r][6] = P2
    b.to_move = P1
    for seed in range(10):
        assert HeuristicAgent(seed=seed).choose_action(b) == 3


def test_two_threats_blocks_one_of_them():
    # Xung đột 4: người 1 thắng được ở cột 0 (xếp chồng) và cột 6 (xếp chồng) -> chặn một trong hai.
    b = board.Board()
    for r in (0, 1, 2):
        b.grid[r][0] = P1
        b.grid[r][6] = P1
    # quân đệm của người 2: không tạo nước thắng nào cho người 2
    for r, c in ((0, 1), (1, 1), (0, 2), (1, 2), (0, 5)):
        b.grid[r][c] = P2
    b.to_move = P2
    agent = HeuristicAgent(seed=0)
    assert agent._winning_columns(b, P2) == []         # người 2 không thắng ngay được
    assert agent._winning_columns(b, P1) == [0, 6]     # người 1 có hai cột thắng
    choices = {HeuristicAgent(seed=s).choose_action(b) for s in range(20)}
    assert choices == {0, 6}                           # luôn chặn một trong hai cột


def test_avoids_giving_opponent_a_win_above():
    # Người 1 có (1,0),(1,1),(1,2); ô (1,3) chưa chơi được vì (0,3) còn trống.
    # Người 2 đặt vào cột 3 sẽ làm (1,3) chơi được và người 1 thắng ngay -> phải tránh cột 3.
    b = board.Board()
    b.grid[0][0], b.grid[0][1], b.grid[0][2] = P2, P1, P2
    b.grid[1][0], b.grid[1][1], b.grid[1][2] = P1, P1, P1
    b.grid[0][4] = P1
    b.grid[0][5], b.grid[0][6] = P2, P2
    b.to_move = P2
    for seed in range(20):
        assert HeuristicAgent(seed=seed).choose_action(b) != 3


def test_returns_legal_action_in_random_games():
    for seed in range(10):
        b = board.Board()
        h, r = HeuristicAgent(seed=seed), RandomAgent(seed=100 + seed)
        while not b.is_terminal():
            agent = h if b.to_move == P1 else r
            col = agent.choose_action(b)
            assert col in b.legal_actions()
            b.step(col)


def test_same_seed_same_moves():
    b = play([3, 2, 4])
    a1, a2 = HeuristicAgent(seed=7), HeuristicAgent(seed=7)
    assert [a1.choose_action(b) for _ in range(20)] == [a2.choose_action(b) for _ in range(20)]


def test_does_not_change_board():
    b = play([3, 3, 2, 4])
    before = ([row[:] for row in b.grid], b.to_move, b.moves_made, list(b.history), b.winner)
    HeuristicAgent(seed=1).choose_action(b)
    assert ([row[:] for row in b.grid], b.to_move, b.moves_made, list(b.history), b.winner) == before


def test_raises_when_game_over():
    b = play([0, 0, 1, 1, 2, 2, 3])
    with pytest.raises(ValueError):
        HeuristicAgent(seed=0).choose_action(b)
