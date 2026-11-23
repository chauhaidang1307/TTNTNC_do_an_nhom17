import argparse
import csv
import sys
import time
import random
from datetime import datetime
from pathlib import Path
from engine.board import Board, PLAYER_1, PLAYER_2
from agents.random_agent import RandomAgent
from agents.heuristic import HeuristicAgent
from agents.alphabeta import AlphaBetaAgent

AGENT_KINDS = ["random", "heuristic", "alphabeta"]

SYMBOLS = {0: ".", PLAYER_1: "X", PLAYER_2: "O"}
# File lưu các seed muốn giữ lại, nằm ở logs/ của dự án (tính từ vị trí file này, không phụ thuộc nơi chạy)
SEED_FILE = Path(__file__).resolve().parent.parent / "logs" / "saved_seeds.csv"


def clear_screen():
    # Mã ANSI: đưa con trỏ về đầu và xóa màn hình (chạy được trên PowerShell/Windows Terminal)
    print("\033[H\033[J", end="")


def draw(board):
    for r in reversed(range(len(board.grid))):  # hàng trên cùng in trước
        print(" ".join(SYMBOLS[v] for v in board.grid[r]))
    print("0 1 2 3 4 5 6")


def make_agent(kind, seed, depth=4, pruning=True, time_limit=None):
    # Tạo agent theo tên loại. depth, pruning, time_limit chỉ dùng cho alphabeta.
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "heuristic":
        return HeuristicAgent(seed=seed)
    if kind == "alphabeta":
        return AlphaBetaAgent(depth=depth, seed=seed, pruning=pruning, time_limit=time_limit)
    raise ValueError(f"Loại agent không hợp lệ: {kind}")


def play_game(agents, delay=0.5, clear=True):
    # agents: dict {PLAYER_1: agent, PLAYER_2: agent}. Trả về Board đã kết thúc.
    board = Board()
    while not board.is_terminal():
        player = board.to_move
        col = agents[player].choose_action(board)
        board.step(col)
        if clear:
            clear_screen()
        print(f"{agents[player].name} ({SYMBOLS[player]}) đi cột {col}  |  nước thứ {board.moves_made}")
        draw(board)
        time.sleep(delay)
    return board


def ask_yes_no(question):
    # Hỏi người dùng; không có bàn phím (chạy tự động, bị chuyển hướng) thì mặc định là KHÔNG lưu.
    try:
        answer = input(f"{question} (y/n): ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes", "c", "co", "có")


def save_seeds(seed1, seed2, agents, board):
    # Ghi thêm một dòng vào logs/saved_seeds.csv (tạo file kèm tiêu đề nếu chưa có).
    # winner: 1 hoặc 2 là người thắng, 0 là hòa.
    SEED_FILE.parent.mkdir(parents=True, exist_ok=True)
    is_new = not SEED_FILE.exists()
    with open(SEED_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["thoi_gian", "agent_1", "seed_1", "agent_2", "seed_2", "winner", "so_nuoc"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            agents[PLAYER_1].name, seed1,
            agents[PLAYER_2].name, seed2,
            board.winner or 0, board.moves_made,
        ])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Demo máy-máy: chọn loại agent cho mỗi bên")
    parser.add_argument("--p1", choices=AGENT_KINDS, default="heuristic", help="agent đi trước (X)")
    parser.add_argument("--p2", choices=AGENT_KINDS, default="random", help="agent đi sau (O)")
    parser.add_argument("--depth", type=int, default=None, help="độ sâu alphabeta (mặc định 4; khi có --time là độ sâu tối đa, mặc định 42)")
    parser.add_argument("--time", type=float, default=None, help="giới hạn giây mỗi nước cho alphabeta (iterative deepening)")
    parser.add_argument("--no-pruning", action="store_true", help="alphabeta tắt cắt tỉa (minimax thuần)")
    parser.add_argument("--seed1", type=int, default=None, help = "seed bỏ trống sẽ chạy random")
    parser.add_argument("--seed2", type=int, default=None, help = "seed bỏ trống sẽ chạy random")
    parser.add_argument("--delay", type=float, default=0.5, help="giây nghỉ giữa các nước")
    parser.add_argument("--no-clear", action="store_true", help="không xóa màn hình")
    args = parser.parse_args()
    if args.seed1 is None:
        args.seed1 = random.randrange(1_000_000)
    if args.seed2 is None:
        args.seed2 = random.randrange(1_000_000)


    pruning = not args.no_pruning
    if args.depth is None:
        args.depth = 4 if args.time is None else 42
    agents = {
        PLAYER_1: make_agent(args.p1, args.seed1, args.depth, pruning, args.time),
        PLAYER_2: make_agent(args.p2, args.seed2, args.depth, pruning, args.time),
    }
    board = play_game(agents, delay=args.delay, clear=not args.no_clear)

    if board.winner is None:
        print("Kết quả: HÒA")
    else:
        print(f"Kết quả: {agents[board.winner].name} ({SYMBOLS[board.winner]}) THẮNG")
    print(f"Seed: {agents[PLAYER_1].name} = {args.seed1}, {agents[PLAYER_2].name} = {args.seed2}")

    if ask_yes_no("Lưu seed của ván này để chạy lại sau?"):
        save_seeds(args.seed1, args.seed2, agents, board)
        print(f"Đã lưu vào {SEED_FILE}")


if __name__ == "__main__":
    main()
