import time

import pytest
from agents.mcts import MCTSAgent
from agents.random_agent import RandomAgent
from arena import tournament
from engine import board

P1, P2 = board.PLAYER_1, board.PLAYER_2


def play(moves):
    b = board.Board()
    for col in moves:
        b.step(col)
    return b


# ---------- tham số và tên ----------

def test_names_and_params():
    a = MCTSAgent(simulations=300, seed=1)
    assert a.name == "MCTS-s300" and a.simulations == 300 and a.time_limit is None
    t = MCTSAgent(simulations=1, seed=1, time_limit=0.05)
    assert t.name == "MCTS-t50ms"


def test_invalid_params():
    with pytest.raises(ValueError):
        MCTSAgent(simulations=0)
    with pytest.raises(ValueError):
        MCTSAgent(time_limit=-1)


def test_raises_when_game_over():
    b = play([0, 1, 0, 1, 0, 1, 0])  # người 1 thắng dọc
    assert b.is_terminal()
    with pytest.raises(ValueError):
        MCTSAgent(simulations=10, seed=0).choose_action(b)


def test_does_not_change_board():
    b = play([3, 3, 2, 4, 2])
    before = ([row[:] for row in b.grid], b.to_move, b.moves_made, list(b.history))
    MCTSAgent(simulations=200, seed=0).choose_action(b)
    assert ([row[:] for row in b.grid], b.to_move, b.moves_made, list(b.history)) == before


def test_returns_legal_action():
    b = play([3, 3, 3, 3, 3, 3])  # cột 3 đã đầy
    a = MCTSAgent(simulations=100, seed=0).choose_action(b)
    assert a in b.legal_actions() and a != 3


# ---------- thống kê của cây ----------

def test_statistics_are_consistent():
    b = play([3, 3, 2, 4])
    a = MCTSAgent(simulations=300, seed=0)
    action = a.choose_action(b)
    assert a.last_nodes == 300
    assert sum(a.last_visits.values()) == 300  # mỗi vòng lặp đi qua đúng một con của gốc
    assert all(0.0 <= v <= 1.0 for v in a.last_values.values())
    assert a.last_visits[action] == max(a.last_visits.values())  # chọn nước có nhiều lần thăm nhất
    assert a.last_depth >= 1
    assert set(a.last_visits) <= set(b.legal_actions())


# ---------- chiến thuật cơ bản (dấu của giá trị đúng cho cả hai bên) ----------

def test_takes_immediate_win_as_player_one():
    b = play([0, 0, 1, 1, 2, 2])  # người 1 đến lượt, cột 3 thắng ngay
    a = MCTSAgent(simulations=500, seed=0)
    assert a.choose_action(b) == 3
    assert a.last_values[3] == 1.0  # nước thắng ngay: mọi mô phỏng thắng cho người đi


def test_takes_immediate_win_as_player_two():
    b = play([1, 0, 1, 0, 2, 0, 5])  # người 2 đến lượt, cột 0 là nước thắng thứ tư
    assert b.to_move == P2
    a = MCTSAgent(simulations=500, seed=0)
    assert a.choose_action(b) == 0
    assert a.last_values[0] == 1.0


def test_blocks_immediate_loss():
    b = play([0, 6, 1, 6, 2])  # người 2 đến lượt, người 1 sắp thắng ở cột 3
    assert MCTSAgent(simulations=1500, seed=0).choose_action(b) == 3


# ---------- tái lập ----------

def test_same_seed_same_move_in_simulation_mode():
    b = play([3, 3, 2, 4, 2, 2])
    m1 = MCTSAgent(simulations=300, seed=7).choose_action(b)
    m2 = MCTSAgent(simulations=300, seed=7).choose_action(b)
    assert m1 == m2


def test_same_seed_same_statistics():
    b = play([3, 3, 2])
    a1, a2 = MCTSAgent(simulations=200, seed=5), MCTSAgent(simulations=200, seed=5)
    a1.choose_action(b)
    a2.choose_action(b)
    assert a1.last_visits == a2.last_visits and a1.last_values == a2.last_values


# ---------- chế độ thời gian ----------

def test_time_mode_runs_at_least_one_iteration():
    a = MCTSAgent(simulations=1, seed=0, time_limit=0)
    assert a.choose_action(board.Board()) in range(7)
    assert a.last_nodes >= 1


def test_time_mode_stops_near_limit_and_ignores_simulations():
    a = MCTSAgent(simulations=1, seed=0, time_limit=0.1)
    t0 = time.perf_counter()
    a.choose_action(board.Board())
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1 + 0.25  # dung sai cho một vòng lặp cuối và máy chậm
    assert a.last_nodes > 1  # chạy nhiều hơn `simulations`=1 vì đang theo thời gian


# ---------- chơi cả ván ----------

def play_vs_random(mcts_first, seed):
    from arena.match import play_game
    m, r = MCTSAgent(simulations=200, seed=seed), RandomAgent(seed=seed + 1000)
    winner, _ = play_game(m, r) if mcts_first else play_game(r, m)
    return (winner == P1) == mcts_first and winner is not None


def test_beats_random_most_of_the_time():
    wins = sum(play_vs_random(i % 2 == 0, i) for i in range(10))
    assert wins >= 9  # ngưỡng lỏng; kiểm tra chương trình chạy đúng, không phải đo sức mạnh


# ---------- tích hợp với arena ----------

@pytest.mark.parametrize("spec, name", [("mcts-s200", "MCTS-s200"), ("mcts-t50", "MCTS-t50ms")])
def test_arena_can_build_mcts(spec, name):
    assert tournament.make_agent(spec, 0).name == name


def test_arena_rejects_nopr_for_mcts():
    with pytest.raises(ValueError):
        tournament.make_agent("mcts-s100-nopr", 0)


# ---------- ván hòa ----------

# Chuỗi 42 nước kết thúc hòa (tìm được bằng ván ngẫu nhiên có seed 344, kiểm tra bên dưới).
DRAW_GAME = [3, 3, 0, 2, 5, 4, 5, 6, 4, 2, 2, 3, 3, 5, 5, 2, 2, 4, 3, 3, 1, 1, 2, 1, 5, 5, 0, 4, 0,
             4, 4, 6, 6, 6, 6, 0, 1, 0, 6, 0, 1, 1]


def test_draw_game_is_really_a_draw():
    assert play(DRAW_GAME).is_terminal() and play(DRAW_GAME).winner is None


def test_draw_is_worth_half_a_point():
    b = play(DRAW_GAME[:-1])  # còn đúng một nước, và nước đó dẫn tới hòa
    assert b.legal_actions() == [DRAW_GAME[-1]]
    a = MCTSAgent(simulations=20, seed=0)
    assert a.choose_action(b) == DRAW_GAME[-1]
    assert a.last_values[DRAW_GAME[-1]] == 0.5
