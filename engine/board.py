import time
import os
ROWS = 6
COLS = 7
EMPTY = 0
PLAYER_1 = 1
PLAYER_2 = 2
WIN_LENGTH = 4
# Bốn hướng kiểm tra thắng, mỗi hướng là (bước hàng, bước cột): ngang, dọc, chéo xuôi, chéo ngược
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]
# Bottom là 0 và Top là ROWS - 1
class Board:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.grid = [[EMPTY] * COLS for _ in range(ROWS)]
        self.to_move = PLAYER_1
        self.moves_made = 0
        self.winner = None
        self.history = []  #List of rows and cols for Undo

    def is_terminal(self): #Have the game win yet?
        return self.winner is not None or self.moves_made == ROWS * COLS #true if winner not None or out of moves

    def result(self,player): #How is it end? Who won or lost? Or it's out of move and draw
        #The Results base on player view, if player 1 win return +1, player 2 win return -1, draw return 0
        if not self.is_terminal():
            raise ValueError("Game haven't ended!")
        if self.winner is None:
            return 0
        return 1 if self.winner == player else -1

    def legal_actions(self):
        if self.is_terminal():
            return []
        top = ROWS - 1
        #Loop through top of each Col and return those Empty spaces
        return [c for c in range(COLS) if self.grid[top][c] == EMPTY]

    def step(self,col):
        if col not in self.legal_actions(): #If the current col not a valid move
            raise ValueError("Think again Chucklenuts!")
        #If it's a valid move
        player = self.to_move #The current player placing
        for row in range(ROWS): #Loop through every rows
            if self.grid[row][col] == EMPTY: #Empty in that row
                self.grid[row][col] = player
                placed_row = row #remember placed row
                break #Only place 1 move in a turn

        self.history.append((col,placed_row))
        self.moves_made += 1 #Raise the moves count
        if self._is_win(placed_row, col, player): #Check winning from 4 directions.
            self.winner = player #Change winner to that player
        self.to_move = PLAYER_2 if self.to_move == PLAYER_1 else PLAYER_1 #Change player after each turn

    def undo(self): #Undo 1 step
        if not self.history:
            raise ValueError("None historic moves")
        col, row = self.history.pop()
        self.grid[row][col] = EMPTY
        self.moves_made -= 1
        self.winner = None
        self.to_move = PLAYER_2 if self.to_move == PLAYER_1 else PLAYER_1

    def clone(self):
        other = Board()
        other.grid = [row[:] for row in self.grid]
        other.to_move = self.to_move
        other.moves_made = self.moves_made
        other.winner = self.winner
        other.history = list(self.history)
        return other

    def _count_one_side(self, row, col, dr, dc, player):
        #Count continuously placed pieces
        n = 0
        r, c = row + dr, col + dc #row = row + direction, col = col + direction
        while 0 <= r < ROWS and 0 <= c < COLS and self.grid[r][c] == player: #Check if it hit the border and same player moves
            n += 1
            r, c = r + dr, c + dc #Continue going.
        return n

    def _is_win(self, row, col, player):
        for dr, dc in DIRECTIONS:
            total = 1 + self._count_one_side(row, col, dr, dc, player) \
                      + self._count_one_side(row, col, -dr, -dc, player)
            if total >= WIN_LENGTH:
                return True
        return False

    def render(self):
        for r in reversed(range(ROWS)):
            print(self.grid[r])


if __name__ == "__main__":
    b = Board()
    steps = [0,0,1,1,2,2,0,3,4,3]
    for s in steps:
        b.step(s)
        time.sleep(0.5)
        os.system('cls')
        b.render()
    for i in range(2):
        if b.result(i+1) == 1:
            print(f"The winner is Player 1")
        elif b.result(i+1) == -1:
            print(f"The winner is Player 2")
            break