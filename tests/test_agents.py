import pytest
from engine import board
from agents.random_agent import RandomAgent


def test_random_returns_legal_action():
    b = board.Board()
    agent = RandomAgent(seed=0)
    for _ in range(200):
        assert agent.choose_action(b) in b.legal_actions()


def test_random_same_seed_same_moves():
    b = board.Board()
    a1, a2 = RandomAgent(seed=123), RandomAgent(seed=123)
    assert [a1.choose_action(b) for _ in range(30)] == [a2.choose_action(b) for _ in range(30)]


def test_random_does_not_change_board():
    b = board.Board()
    b.step(3)
    before = ([row[:] for row in b.grid], b.to_move, b.moves_made, list(b.history))
    RandomAgent(seed=1).choose_action(b)
    assert ([row[:] for row in b.grid], b.to_move, b.moves_made, list(b.history)) == before


def test_random_only_picks_non_full_columns():
    b = board.Board()
    for _ in range(board.ROWS):  # đầy cột 0 (xen kẽ hai người, không ai thắng)
        b.step(0)
    agent = RandomAgent(seed=5)
    assert all(agent.choose_action(b) != 0 for _ in range(200))


def test_random_raises_when_game_over():
    b = board.Board()
    for col in [0, 0, 1, 1, 2, 2, 3]:  # người 1 thắng ngang
        b.step(col)
    with pytest.raises(ValueError):
        RandomAgent(seed=0).choose_action(b)


def test_random_vs_random_always_finishes():
    for seed in range(20):
        b = board.Board()
        p1, p2 = RandomAgent(seed=seed), RandomAgent(seed=1000 + seed)
        while not b.is_terminal():
            agent = p1 if b.to_move == board.PLAYER_1 else p2
            b.step(agent.choose_action(b))
        assert b.is_terminal()
        assert b.moves_made <= board.ROWS * board.COLS
