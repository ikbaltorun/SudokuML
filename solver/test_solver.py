import numpy as np
import tensorflow as tf
import random

# =========================================================
# MODELİ YÜKLE
# =========================================================
MODEL_PATH = "../model/sudoku_ml_model.keras" 
model = tf.keras.models.load_model(MODEL_PATH)


# =========================================================
# SUDOKU KONTROL & OLUŞTURMA
# =========================================================
def is_valid(board, row, col, num):
    if num in board[row]: return False
    if num in board[:, col]: return False
    box_row, box_col = (row // 3) * 3, (col // 3) * 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r][c] == num: return False
    return True

def solve_sudoku(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                numbers = list(range(1, 10))
                random.shuffle(numbers)
                for num in numbers:
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku(board): return True
                        board[row][col] = 0
                return False
    return True

def generate_solution():
    board = np.zeros((9, 9), dtype=np.int8)
    solve_sudoku(board)
    return board

def create_puzzle(solution, empty_cells=45):
    puzzle = solution.copy()
    positions = list(range(81))
    random.shuffle(positions)
    for position in positions[:empty_cells]:
        puzzle[position // 9][position % 9] = 0
    return puzzle


# =========================================================
# YENİ CNN TAHMİN MANTIĞI
# =========================================================
def get_candidates(board, row, col):
    return [num for num in range(1, 10) if is_valid(board, row, col, num)]

def predict_board(board):
    # Yeni One-Hot CNN yapımız
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


# =========================================================
# ML + BACKTRACKING ÇÖZÜCÜ
# =========================================================
def solve_with_ml(board):
    steps = 0

    def backtrack():
        nonlocal steps
        steps += 1

        if not np.any(board == 0): return True

        # Modeli bir kez çağır
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
# 100 SUDOKU TESTİ (DÜZELTİLMİŞ NOTLANDIRMA)
# =========================================================
if __name__ == "__main__":
    TOTAL_TESTS = 100
    successful = 0
    failed = 0

    print("\n========================================")
    print("ML SUDOKU TESTİ BAŞLIYOR (CNN MİMARİSİ)")
    print("========================================\n")

    for test_number in range(1, TOTAL_TESTS + 1):
        solution = generate_solution()
        puzzle = create_puzzle(solution, empty_cells=45)
        board = puzzle.copy()

        # Algoritma çözümü dener
        success, steps = solve_with_ml(board)
        
        # YENİ KONTROL MANTIĞI:
        # Eğer 'success' True döndüyse, algoritma zaten kuralları ihlal etmeyen 
        # (is_valid) bir çözüm bulmuş ve tahtada hiç '0' kalmamış demektir.
        if success:
            successful += 1
            print(f"Test {test_number:3d} → ✅ BAŞARILI ({steps} adım)")
        else:
            failed += 1
            print(f"Test {test_number:3d} → ❌ BAŞARISIZ")

    accuracy = (successful / TOTAL_TESTS) * 100

    print("\n========================================")
    print("TEST SONUCU")
    print("========================================")
    print(f"Toplam test : {TOTAL_TESTS}")
    print(f"Başarılı    : {successful}")
    print(f"Başarısız   : {failed}")
    print(f"Başarı      : %{accuracy:.2f}")
    print("========================================")