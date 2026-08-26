import numpy as np
import random

# AYARLAR (Veri seti CNN için büyütüldü)
TOTAL_SUDOKUS = 100000 
TRAIN_SUDOKUS = 80000 # 80000 Sudoku eğitim verisi için
TEST_SUDOKUS = 20000 # 20000 Sudoku test verisi için
OUTPUT_PATH = "sudoku_ml_dataset.npz"

def is_valid(board, row, col, num): #sudoku olup olmadığı kontrolü
    if num in board[row]: return False
    if num in board[:, col]: return False
    box_row, box_col = (row // 3) * 3, (col // 3) * 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r][c] == num: return False
    return True

def solve_sudoku(board): #sudokunun çözülüp çözülmediği kontrolü
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                numbers = list(range(1, 10))
                random.shuffle(numbers)
                for num in numbers:
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku(board): return True #backtracking algoritması
                        board[row][col] = 0
                return False
    return True

def generate_solution(): 
    board = np.zeros((9, 9), dtype=np.int8)
    solve_sudoku(board)
    return board

def create_puzzle(solution, empty_cells): 
    puzzle = solution.copy()
    positions = list(range(81))
    random.shuffle(positions)
    for position in positions[:empty_cells]:
        puzzle[position // 9][position % 9] = 0
    return puzzle

def create_dataset(number_of_sudokus): #verinin işlenebilmesi için dönüştürülmesi
    X, y = [], []
    print(f"{number_of_sudokus} Sudoku üretiliyor...")
    
    for i in range(number_of_sudokus):
        solution = generate_solution() #önce sudokular oluşturulur
        empty_cells = random.randint(40, 50) #40-50 arası random bi şekilde silinir
        puzzle = create_puzzle(solution, empty_cells) #puzzle elde edilir

        # Tahtayı (9, 9, 10) boyutunda One-Hot Encoding yapıyoruz
        puzzle_encoded = np.eye(10)[puzzle]
        
        # Hedef veriyi (9, 9) boyutunda ayarlıyoruz (sınıflar 0-8 arası)
        target = solution - 1

        X.append(puzzle_encoded)
        y.append(target)

        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{number_of_sudokus}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

print("TRAIN DATASET") 
X_train, y_train = create_dataset(TRAIN_SUDOKUS)
#80k x train eğitim için boşluklu sudoku
#580k y train cevap anahtarı

print("TEST DATASET") 
X_test, y_test = create_dataset(TEST_SUDOKUS)
#20k x test modelin görmediği boşluklu sudoku
#20k y test doğru çözümler

np.savez_compressed(OUTPUT_PATH, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test) 

print("\nDataset kaydedildi:", OUTPUT_PATH)