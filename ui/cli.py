import argparse
import sys
import time

from engine.board import Board, PLAYER_1, PLAYER_2
from agents.random_agent import RandomAgent

SYMBOLS = {0: ".", PLAYER_1: "X", PLAYER_2: "O"}


def clear_screen():
    # Mã ANSI: đưa con trỏ về đầu và xóa màn hình (chạy được trên PowerShell/Windows Terminal)
    print("\033[H\033[J", end="")


def draw(board):
    for r in reversed(range(len(board.grid))):  # hàng trên cùng in trước
        print(" ".join(SYMBOLS[v] for v in board.grid[r]))
    print("0 1 2 3 4 5 6")


def play_game(agents, delay=0.5, clear=True):
    # agents: dict {PLAYER_1: agent, PLAYER_2: agent}. Trả về Board đã kết thúc.
    board = Board()
    while not board.is_terminal():
        player = board.to_move
        col = agents[player].choose_action(board)
        board.step(col)
        if clear:
            clear_screen()
        print(f"Người {player} ({SYMBOLS[player]}) đi cột {col}  |  nước thứ {board.moves_made}")
        draw(board)
        time.sleep(delay)
    return board


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Demo máy-máy: Random đấu Random")
    parser.add_argument("--seed1", type=int, default=1)
    parser.add_argument("--seed2", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.5, help="giây nghỉ giữa các nước")
    parser.add_argument("--no-clear", action="store_true", help="không xóa màn hình")
    args = parser.parse_args()

    agents = {PLAYER_1: RandomAgent(seed=args.seed1), PLAYER_2: RandomAgent(seed=args.seed2)}
    board = play_game(agents, delay=args.delay, clear=not args.no_clear)

    if board.winner is None:
        print("Kết quả: HÒA")
    else:
        print(f"Kết quả: người {board.winner} ({SYMBOLS[board.winner]}) THẮNG")
    print(f"Seed: người 1 = {args.seed1}, người 2 = {args.seed2}")


if __name__ == "__main__":
    main()
