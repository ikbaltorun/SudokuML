import numpy as np
import tensorflow as tf
import time

# =========================================================
# MODELİ YÜKLE
# =========================================================
MODEL_PATH = "../model/sudoku_ml_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)


# =========================================================
# YARDIMCI VE ÇÖZÜM FONKSİYONLARI
# =========================================================
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

def predict_board(board):
    encoded = np.eye(10)[board].astype(np.float32)
    encoded = encoded.reshape(1, 9, 9, 10)
    probabilities = model.predict(encoded, verbose=0)[0]
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
                
                cell_max_prob = max([probabilities[row][col][num - 1] for num in candidates])
                if cell_max_prob > highest_prob:
                    highest_prob = cell_max_prob
                    best_row, best_col = row, col
                    best_moves = [(probabilities[row][col][num - 1], num) for num in candidates]
                    best_moves.sort(key=lambda x: x[0], reverse=True)
                    
    return best_row, best_col, best_moves

def solve_with_ml(board):
    steps = 0
    def backtrack():
        nonlocal steps
        steps += 1
        if not np.any(board == 0): return True
        probabilities = predict_board(board)
        row, col, moves = get_best_cell_moves(board, probabilities)
        if moves is None: return False
        for probability, number in moves:
            board[row][col] = number
            if backtrack(): return True
            board[row][col] = 0
        return False

    success = backtrack()
    return success, steps


# =========================================================
# DÜNYACA ÜNLÜ EN ZOR SUDOKU BULMACALARI
# =========================================================
# 1. AI Escargot (Arto Inkala tarafından tasarlanan dünyanın en zor bulmacası)
ai_escargot = np.array([
    [1, 0, 0, 0, 0, 7, 0, 9, 0],
    [0, 3, 0, 0, 2, 0, 0, 0, 8],
    [0, 0, 9, 6, 0, 0, 5, 0, 0],
    [0, 0, 5, 3, 0, 0, 9, 0, 0],
    [0, 1, 0, 0, 8, 0, 0, 0, 2],
    [6, 0, 0, 0, 0, 4, 0, 0, 0],
    [3, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 4, 0, 0, 0, 0, 0, 0, 7],
    [0, 0, 7, 0, 0, 0, 3, 0, 0]
])

# 2. 17-Clue Minimal Sudoku (Matematiksel olarak çözülebilir en az ipuçlu Sudoku)
minimal_17 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 3],
    [0, 0, 2, 3, 0, 0, 4, 0, 0],
    [0, 0, 1, 8, 0, 0, 0, 0, 5],
    [0, 6, 0, 0, 7, 0, 8, 0, 0],
    [0, 0, 0, 0, 0, 9, 0, 0, 0],
    [0, 0, 8, 5, 0, 0, 0, 0, 0],
    [9, 0, 0, 0, 4, 0, 5, 0, 0],
    [4, 7, 0, 0, 0, 6, 0, 0, 0]
])

challenges = [
    ("AI Escargot (Dünyanın En Zor Sudokusu)", ai_escargot),
    ("17-Clue Minimal Sudoku (En Az İpuçlu)", minimal_17)
]

print("\n=======================================================")
print("🔥 EKSTREM SUDOKU MEYDAN OKUMA TESTİ BAŞLIYOR 🔥")
print("=======================================================\n")

for name, puzzle in challenges:
    print(f"👉 Test Ediliyor: {name}")
    board = puzzle.copy()
    
    start_time = time.time()
    success, steps = solve_with_ml(board)
    elapsed = (time.time() - start_time) * 1000  # ms
    
    if success:
        print(f"   Sonuç : ✅ BAŞARILI!")
        print(f"   Adım  : {steps} adım")
        print(f"   Süre  : {elapsed:.2f} ms\n")
    else:
        print(f"   Sonuç : ❌ BAŞARISIZ\n")

print("=======================================================")