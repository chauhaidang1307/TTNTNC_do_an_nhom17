import time

from engine.board import Board, PLAYER_1, PLAYER_2


def play_game(first, second):
    # first đi trước (PLAYER_1), second đi sau (PLAYER_2).
    # Trả về (winner: PLAYER_1 / PLAYER_2 / None nếu hòa, danh sách bản ghi từng nước).
    agents = {PLAYER_1: first, PLAYER_2: second}
    board = Board()
    moves = []
    while not board.is_terminal():
        player = board.to_move
        agent = agents[player]
        t0 = time.perf_counter()
        col = agent.choose_action(board)
        dt = time.perf_counter() - t0
        moves.append(dict(
            ply=board.moves_made + 1,
            player=player,
            agent=agent.name,
            col=col,
            time_ms=round(dt * 1000, 4),
            nodes=getattr(agent, "last_nodes", ""),
            depth=getattr(agent, "last_depth", ""),
        ))
        board.step(col)  # nước bất hợp lệ sẽ bị engine từ chối bằng ngoại lệ
    return board.winner, moves
