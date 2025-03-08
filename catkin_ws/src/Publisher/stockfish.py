
import chess
import chess.engine

def play_chess_with_stockfish():
    # Path to your Stockfish binary (update this with your Stockfish installation path)
    stockfish_path = '/usr/games/stockfish'

    # Initialize the Stockfish engine
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        board = chess.Board()

        while not board.is_game_over():
            print(board)
            
            # Get the user's move
            user_move_uci = input("Enter your move (in UCI notation, e.g., 'e2e4'): ")
            
            # Validate and make the user's move
            try:
                user_move = chess.Move.from_uci(user_move_uci)
                if user_move in board.legal_moves:
                    board.push(user_move)
                else:
                    print("Invalid move. Try again.")
                    continue
            except ValueError:
                print("Invalid move format. Try again.")
                continue
            
            print(f"You played: {user_move_uci}")
            
            # Get the best move from Stockfish in response to the user's move
            result = engine.play(board, chess.engine.Limit(time=0.1))
            
            print(f"Stockfish suggests: {result.move}")
            
            # Make Stockfish's move
            board.push(result.move)

    print("Game over")
    print(board.result())

if __name__ == "__main__":
    play_chess_with_stockfish()
'''
import tkinter as tk
from PIL import Image, ImageTk
import chess
import chess.svg
import io

class ChessUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess UI")

        self.board = chess.Board()
        self.board_canvas = tk.Canvas(self.root, width=400, height=400)
        self.board_canvas.pack()
        self.update_board()

        self.status_label = tk.Label(self.root, text="Your move")
        self.status_label.pack()

        self.stockfish = chess.engine.SimpleEngine.popen_uci('/usr/games/stockfish')
        self.play_stockfish_move()

        self.board_canvas.bind("<Button-1>", self.on_click)

    def on_click(self, event):
        if not self.board.is_game_over():
            col = event.x // 50
            row = 7 - (event.y // 50)

            square = chess.square(col, row)
            moves = [move for move in self.board.legal_moves if move.from_square == square]
            if moves:
                self.board.push(moves[0])
                self.update_board()
                self.status_label.config(text="Thinking...")
                self.root.after(100, self.play_stockfish_move)
    
    def play_stockfish_move(self):
        if not self.board.is_game_over():
            result = self.stockfish.play(self.board, chess.engine.Limit(time=0.1))
            self.board.push(result.move)
            self.update_board()
            self.status_label.config(text="Your move")

    def update_board(self):
        self.board_canvas.delete("all")
        svgboard = chess.svg.board(self.board)
        svgboard = svgboard.encode("utf-8")
        board_image = Image.open(io.BytesIO(svgboard))
        self.board_image = ImageTk.PhotoImage(board_image)

        self.board_canvas.create_text(200, 200, text="Your Move", fill="blue", font=("Arial", 14))
        self.board_canvas.create_text(200, 20, text="Stockfish", fill="red", font=("Arial", 14))
        self.board_canvas.create_image(0, 0, anchor="nw", image=self.board_image)

if __name__ == "__main__":
    root = tk.Tk()
    chess_ui = ChessUI(root)
    root.mainloop()

'''
