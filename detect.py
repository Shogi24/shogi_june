from ultralytics import YOLO
import cv2
import numpy as np
import shogi
from collections import defaultdict

# --- Load YOLOv8 Model ---
model = YOLO("best.pt")  # Your trained Shogi piece detection model

# --- Load and Process the Image ---
image_path = "shogi_board.jpg"
img = cv2.imread(image_path)
results = model(img)[0]

# --- Get image size and set 9x9 grid ---
height, width = img.shape[:2]
cell_w = width / 9
cell_h = height / 9

def get_shogi_grid(x_center, y_center):
    col = int(x_center // cell_w)  # 0–8
    row = int(y_center // cell_h)  # 0–8
    # Convert to Shogi format (9 to 1, a to i)
    shogi_file = 9 - col
    shogi_rank = chr(ord('a') + row)  # 'a' = top row
    return f"{shogi_file}{shogi_rank}"

# --- Label Map (adjust based on your training classes) ---
label_map = {
    0: "P", 1: "+P", 2: "L", 3: "+L",
    4: "N", 5: "+N", 6: "S", 7: "+S",
    8: "G", 9: "K", 10: "B", 11: "+B",
    12: "R", 13: "+R"
}

# --- Log Piece Positions ---
board_positions = defaultdict(str)

for box in results.boxes:
    cls_id = int(box.cls[0].item())
    label = label_map.get(cls_id, "?")

    # Get center point
    x1, y1, x2, y2 = box.xyxy[0]
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    pos = get_shogi_grid(x_center, y_center)
    board_positions[pos] = label

# --- Display Logged Positions ---
print("📋 Detected Shogi Pieces and Positions:")
for pos in sorted(board_positions):
    print(f"{pos}: {board_positions[pos]}")

# Optional: SFEN string builder placeholder (non-trivial if some positions missing)
# You can reconstruct SFEN from board_positions if needed

# --- If you want to manually set pieces into shogi.Board ---
board = shogi.Board()
board.clear()

# Mapping shogi coords like "7f" to index
def shogi_pos_to_index(pos):
    file = int(pos[0])
    rank = ord(pos[1]) - ord('a') + 1
    return shogi.SQUARES[f"{file}{rank}"]

# Simple insertion example
for pos, piece in board_positions.items():
    try:
        square = shogi.SQUARES[pos]
        board.set_piece_at(square, shogi.Piece(shogi.BLACK, shogi.PIECE_SYMBOLS.index(piece.lower())))
    except:
        pass  # Skip unrecognized pieces or bad positions

# --- Print Final Board ---
print("\n♟️ Final Shogi Board:")
print(board)

