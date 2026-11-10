class Agent:
    # Giao diện chung của mọi tác tử: nhận một Board, trả về một cột hợp lệ.
    # Tác tử chỉ ĐỌC board, không được gọi step() trực tiếp trên board của arena
    # (nếu cần thử nước đi thì dùng board.clone()).
    name = "Agent"

    def choose_action(self, board):
        raise NotImplementedError
