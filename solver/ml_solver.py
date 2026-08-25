import numpy as np
import tensorflow as tf

MODEL_PATH = "../model/sudoku_ml_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

def is_valid(board, row, col, num):
    if num in board[row]: return False
    if num in board[:, col]: return False
    box_row, box_col = (row // 3) * 3, (col // 3) * 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r][c] == num: return False
    return True

def get_candidates(board, row, col):
    return [num for num in range(1, 10) if is_valid(board, row, col, num)]

# BÜTÜN TAHTAYI TEK SEFERDE TAHMİN ET
def predict_board(board):
    encoded = np.eye(10)[board].astype(np.float32) # (9, 9, 10)
    encoded = encoded.reshape(1, 9, 9, 10)
    probabilities = model.predict(encoded, verbose=0)[0] # Çıktı: (9, 9, 9)
    return probabilities

def get_best_cell_moves(board, probabilities):
    best_row, best_col = -1, -1
    highest_prob = -1
    best_moves = []

    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                candidates = get_candidates(board, row, col)
                if not candidates: return None, None, None
                
                # Bu hücre için modelin adaylara verdiği en yüksek güven skoru
                cell_max_prob = max([probabilities[row][col][num - 1] for num in candidates])
                
                if cell_max_prob > highest_prob:
                    highest_prob = cell_max_prob
                    best_row, best_col = row, col
                    best_moves = [(probabilities[row][col][num - 1], num) for num in candidates]
                    best_moves.sort(key=lambda x: x[0], reverse=True)
                    
    return best_row, best_col, best_moves

steps = 0
def solve(board, depth=0):
    global steps
    steps += 1

    if not np.any(board == 0): return True

    # ÖNEMLİ: Modeli her hamlede tahtanın geneli için sadece BİR KERE çağırıyoruz!
    probabilities = predict_board(board)
    row, col, moves = get_best_cell_moves(board, probabilities)

    if moves is None: return False

    for probability, number in moves:
        board[row][col] = number
        print(f"Derinlik {depth} | ({row + 1},{col + 1}) = {number} | ML %{probability * 100:.2f}")

        if solve(board, depth + 1): return True

        board[row][col] = 0
        print(f"↩ Geri dönüldü: ({row + 1},{col + 1}) = {number}")

    return False

if __name__ == "__main__":
    sudoku = np.array([
        [0, 0, 0, 2, 6, 0, 7, 0, 1],
        [6, 8, 0, 0, 7, 0, 0, 9, 0],
        [1, 9, 0, 0, 0, 4, 5, 0, 0],
        [8, 2, 0, 1, 0, 0, 0, 4, 0],
        [0, 0, 4, 6, 0, 2, 9, 0, 0],
        [0, 5, 0, 0, 0, 3, 0, 2, 8],
        [0, 0, 9, 3, 0, 0, 0, 7, 4],
        [0, 4, 0, 0, 5, 0, 0, 3, 6],
        [7, 0, 3, 0, 1, 8, 0, 0, 0]
    ], dtype=np.int8)

    print("\nML + BACKTRACKING BAŞLIYOR\n")
    solved = solve(sudoku)
    
    if solved:
        print("\nSUDOKU ÇÖZÜLDÜ!")
        for r in sudoku: print(r)
        print(f"\nToplam arama adımı: {steps}")