import streamlit as st
import numpy as np
import tensorflow as tf
import random
import time

# Sayfa Yapılandırması
st.set_page_config(
    page_title="CNN Sudoku Solver",
    page_icon="🧩",
    layout="centered"
)

# Modelin Yüklenmesi
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/sudoku_ml_model.keras")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu! Hata: {e}")

# =========================================================
# SUDOKU KURALLARI VE YARDIMCI FONKSİYONLAR
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

# =========================================================
# WEB ARAYÜZÜ (STREAMLIT) TASARIMI
# =========================================================
st.title("🧩 CNN Destekli Hibrit Sudoku Çözücü")
st.markdown("Yapay zekanın hamleleri nasıl adım adım denediğini izleyin.")

# Oturum Durumu (State) Yönetimi
if "board" not in st.session_state:
    sol = generate_solution()
    st.session_state.solution = sol
    st.session_state.puzzle = create_puzzle(sol, empty_cells=45)
    st.session_state.current_board = st.session_state.puzzle.copy()
    st.session_state.status_message = ""

col1, col2 = st.columns(2)
with col1:
    if st.button("🎲 Yeni Bulmaca Üret", use_container_width=True):
        sol = generate_solution()
        st.session_state.solution = sol
        st.session_state.puzzle = create_puzzle(sol, empty_cells=45)
        st.session_state.current_board = st.session_state.puzzle.copy()
        st.session_state.status_message = ""
        st.rerun()

with col2:
    solve_clicked = st.button("✨ Yapay Zeka ile Çöz", use_container_width=True)

# Sudoku Tahtasını Çizen Fonksiyon
def render_board(board, active_cell=None, trial_num=None):
    html = """
    <style>
    .sudoku-table {
        margin: 20px auto;
        border-collapse: collapse;
        border: 3px solid #ffffff;
        background-color: #0e1117;
    }
    .sudoku-cell {
        width: 50px;
        height: 50px;
        text-align: center;
        vertical-align: middle;
        font-size: 22px;
        font-weight: 600;
        border: 1px solid #333333;
    }
    .border-top-thick { border-top: 3px solid #ffffff !important; }
    .border-left-thick { border-left: 3px solid #ffffff !important; }
    .border-bottom-thick { border-bottom: 3px solid #ffffff !important; }
    .border-right-thick { border-right: 3px solid #ffffff !important; }
    </style>
    <table class="sudoku-table">
    """
    
    for r_idx, row in enumerate(board):
        html += "<tr>"
        for c_idx, val in enumerate(row):
            classes = ["sudoku-cell"]
            if r_idx in [0, 3, 6]: classes.append("border-top-thick")
            if r_idx == 8: classes.append("border-bottom-thick")
            if c_idx in [0, 3, 6]: classes.append("border-left-thick")
            if c_idx == 8: classes.append("border-right-thick")
            
            bg_color = "transparent"
            text_color = "#ffffff"
            display_val = str(val) if val != 0 else ""
            
            if active_cell and active_cell == (r_idx, c_idx):
                bg_color = "rgba(234, 179, 8, 0.3)"
                if trial_num:
                    display_val = str(trial_num)
                    text_color = "#facc15"
            
            html += f"<td class='{' '.join(classes)}' style='background-color: {bg_color};'><span style='color: {text_color};'>{display_val}</span></td>"
        html += "</tr>"
    html += "</table>"
    return html

# Tahta Konteynırı
board_container = st.empty()
status_container = st.empty()

board_container.markdown(render_board(st.session_state.current_board), unsafe_allow_html=True)

if st.session_state.status_message:
    status_container.info(st.session_state.status_message)

# Animasyonlu Çözüm Mantığı
if solve_clicked:
    board_to_solve = st.session_state.current_board.copy()
    state = {"steps": 0, "success": False}

    def animated_backtrack():
        def _backtrack():
            state["steps"] += 1
            if not np.any(board_to_solve == 0):
                return True

            probabilities = predict_board(board_to_solve)
            row, col, moves = get_best_cell_moves(board_to_solve, probabilities)

            if moves is None: return False

            for probability, number in moves:
                board_container.markdown(render_board(board_to_solve, active_cell=(row, col), trial_num=number), unsafe_allow_html=True)
                time.sleep(0.03)

                board_to_solve[row][col] = number
                if _backtrack(): return True
                board_to_solve[row][col] = 0

            return False

        return _backtrack()

    with st.spinner("Yapay zeka bulmacayı çözüyor..."):
        solved_status = animated_backtrack()

    if solved_status:
        st.session_state.current_board = board_to_solve
        board_container.markdown(render_board(st.session_state.current_board), unsafe_allow_html=True)
        status_container.success(f"🎯 Bulmaca başarıyla çözüldü! Toplam Adım: {state['steps']}")
    else:
        status_container.error("Bulmaca çözülemedi.")