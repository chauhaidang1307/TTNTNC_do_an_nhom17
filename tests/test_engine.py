import pytest
from engine import board

# Chuỗi 42 nước kết thúc hòa, tìm bằng random có seed 344 (chỉ để lấy dữ liệu test)
DRAW_MOVES = [3, 3, 0, 2, 5, 4, 5, 6, 4, 2, 2, 3, 3, 5, 5, 2, 2, 4, 3, 3, 1,
              1, 2, 1, 5, 5, 0, 4, 0, 4, 4, 6, 6, 6, 6, 0, 1, 0, 6, 0, 1, 1]
P1_WIN_MOVES = [0, 0, 1, 1, 2, 2, 3]
P2_WIN_MOVES = [0,0,1,1,2,2,1,3,4,3]

def play(moves):
    b = board.Board()
    for col in moves:
        b.step(col)
    return b


def test_draw_after_42_moves():
    b = play(DRAW_MOVES)
    assert b.moves_made == 42
    assert b.winner is None
    assert b.is_terminal()
    assert b.legal_actions() == []
    assert b.result(board.PLAYER_1) == 0
    assert b.result(board.PLAYER_2) == 0


def test_not_terminal_at_start():
    assert not board.Board().is_terminal()


def test_result_P1():
    b = play(P1_WIN_MOVES)
    assert b.is_terminal()
    assert b.result(board.PLAYER_1) == 1
    assert b.result(board.PLAYER_2) == -1

def test_result_P2():
    b = play(P2_WIN_MOVES)
    assert b.is_terminal()
    assert b.result(board.PLAYER_1) == -1
    assert b.result(board.PLAYER_2) == 1


def test_result_before_end_raises():
    with pytest.raises(ValueError):
        board.Board().result(board.PLAYER_1)


def test_step_after_win_raises():
    b = play(P1_WIN_MOVES)
    with pytest.raises(ValueError):
        b.step(4)


def test_undo_restores_state():
    b = play([3, 3, 4])
    b.undo()
    assert b.moves_made == 2
    assert b.to_move == board.PLAYER_1
    assert b.grid[0][4] == board.EMPTY
    assert b.grid[1][3] == board.PLAYER_2


def test_undo_after_win_clears_winner():
    b = play(P1_WIN_MOVES)
    b.undo()
    assert b.winner is None
    assert not b.is_terminal()
    assert b.to_move == board.PLAYER_1


def test_undo_on_empty_board_raises():
    with pytest.raises(ValueError):
        board.Board().undo()


def test_undo_all_returns_to_start():
    b = play(DRAW_MOVES)
    for _ in range(42):
        b.undo()
    assert b.grid == board.Board().grid
    assert b.moves_made == 0 and b.to_move == board.PLAYER_1


def test_clone_is_independent():
    b = play([3, 3])
    c = b.clone()
    c.step(0)
    assert b.moves_made == 2
    assert b.grid[0][0] == board.EMPTY
    assert c.moves_made == 3
    c.undo()
    assert c.grid == b.grid and c.history == b.history


def test_reset():
    b = play(P1_WIN_MOVES)
    b.reset()
    assert b.grid == board.Board().grid
    assert b.winner is None and b.moves_made == 0 and b.history == []
