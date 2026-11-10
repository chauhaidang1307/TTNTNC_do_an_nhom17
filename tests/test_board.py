from engine import board
import pytest

def test_new_board():
    b = board.Board()
    assert b.legal_actions() == [0,1,2,3,4,5,6]
    assert b.to_move == board.PLAYER_1


def test_grid_shape():
    b = board.Board()
    assert len(b.grid) == board.ROWS
    assert all(len(row) == board.COLS for row in b.grid)

def test_rows_are_independent():
    b = board.Board()
    b.grid[0][0] = board.PLAYER_1
    assert b.grid[1][0] == board.EMPTY

def test_step_drops_to_bottom_and_switches_turn():
    b = board.Board()
    b.step(3)
    assert b.grid[0][3] == board.PLAYER_1
    assert b.to_move == board.PLAYER_2
    assert b.moves_made == 1


def test_step_stacks_pieces():
    b = board.Board()
    b.step(3)
    b.step(3)
    assert b.grid[0][3] == board.PLAYER_1
    assert b.grid[1][3] == board.PLAYER_2


def test_full_column_is_illegal():
    b = board.Board()
    for _ in range(board.ROWS):
        b.step(0)
    assert 0 not in b.legal_actions()
    with pytest.raises(ValueError):
        b.step(0)


def test_out_of_range_column_raises():
    b = board.Board()
    with pytest.raises(ValueError):
        b.step(7)
    with pytest.raises(ValueError):
        b.step(-1)


def test_failed_step_does_not_change_state():
    b = board.Board()
    with pytest.raises(ValueError):
        b.step(9)
    assert b.moves_made == 0
    assert b.to_move == board.PLAYER_1

def test_horizontal_win():
    b = board.Board()
    for col in [0,0,1,1,2,2,3]:
        b.step(col)
    assert b.winner == board.PLAYER_1

def test_three_in_a_row_is_not_a_win():
    b = board.Board()
    for col in [0,0,1,1,2,2]:
        b.step(col)
    assert b.winner is None 

def test_vertical_win():
    b = board.Board()
    for col in [0, 1, 0, 1, 0, 1, 0]:
        b.step(col)
    assert b.winner == board.PLAYER_1


def test_diagonal_up_right_win():
    # Dựng thẳng vào grid: người 1 ở (0,0),(1,1),(2,2); các ô đệm là người 2.
    b = board.Board()
    b.grid[0][0] = board.PLAYER_1
    b.grid[0][1], b.grid[1][1] = board.PLAYER_2, board.PLAYER_1
    b.grid[0][2], b.grid[1][2], b.grid[2][2] = board.PLAYER_2, board.PLAYER_2, board.PLAYER_1
    b.grid[0][3], b.grid[1][3], b.grid[2][3] = board.PLAYER_2, board.PLAYER_2, board.PLAYER_2
    b.step(3)  # người 1 đặt vào (3,3)
    assert b.winner == board.PLAYER_1


def test_diagonal_up_left_win():
    b = board.Board()
    b.grid[0][6] = board.PLAYER_1
    b.grid[0][5], b.grid[1][5] = board.PLAYER_2, board.PLAYER_1
    b.grid[0][4], b.grid[1][4], b.grid[2][4] = board.PLAYER_2, board.PLAYER_2, board.PLAYER_1
    b.grid[0][3], b.grid[1][3], b.grid[2][3] = board.PLAYER_2, board.PLAYER_2, board.PLAYER_2
    b.step(3)  # người 1 đặt vào (3,3)
    assert b.winner == board.PLAYER_1


def test_horizontal_blocked_is_not_a_win():
    # Hàng đáy: 1 2 1 1 [1]  -> bên trái chỉ đếm được 2 vì cột 1 là quân đối thủ
    b = board.Board()
    b.grid[0][0], b.grid[0][1] = board.PLAYER_1, board.PLAYER_2
    b.grid[0][2], b.grid[0][3] = board.PLAYER_1, board.PLAYER_1
    b.step(4)
    assert b.winner is None


def test_win_joins_two_groups():
    # 111 _ 111: đặt vào giữa ra 7 quân liên tiếp, điều kiện phải là >= 4
    b = board.Board()
    for c in (0, 1, 2, 4, 5, 6):
        b.grid[0][c] = board.PLAYER_1
    b.step(3)
    assert b.winner == board.PLAYER_1
