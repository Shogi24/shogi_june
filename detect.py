from ultralytics import YOLO
import cv2
import numpy as np
import shogi

# --- Load YOLOv8 Model ---
model = YOLO("polyk (b2-100).pt")  # Trained model path

# --- Load and Process the Image ---
image_path = "sample.jpg"
img = cv2.imread(image_path)
height, width = img.shape[:2]
results = model(img)[0]

# --- Grid Settings for 9x9 Shogi Board ---
cell_w = width / 9
cell_h = height / 9

# --- Label Map for Class Index to Shogi Notation ---
label_map = {
    0: "P", 1: "+P", 2: "L", 3: "+L",
    4: "N", 5: "+N", 6: "S", 7: "+S",
    8: "G", 9: "K", 10: "B", 11: "+B",
    12: "R", 13: "+R"
}

# --- Initialize 9x9 Board Grid ---
# Each cell will either be '' or a piece symbol
grid = [['' for _ in range(9)] for _ in range(9)]

# --- Process YOLO Detections ---
for box in results.boxes:
    cls_id = int(box.cls[0].item())
    label = label_map.get(cls_id, "?")

    # Get center point of the box
    x1, y1, x2, y2 = box.xyxy[0]
    x_center = float((x1 + x2) / 2)
    y_center = float((y1 + y2) / 2)

    # Determine board grid row, col
    col = int(x_center // cell_w)  # 0–8
    row = int(y_center // cell_h)  # 0–8

    if 0 <= row < 9 and 0 <= col < 9:
        grid[row][col] = label

# --- Log Detected Board ---
print("📋 Shogi Board (Top-down View):")
for r in range(9):
    row_data = []
    for c in range(9):
        cell = grid[r][c] if grid[r][c] else "."
        row_data.append(cell)
    print(f"Row {chr(ord('a') + r)}: {' '.join(row_data)}")

# --- Initialize a shogi.Board and Clear it ---
board = shogi.Board()
board.clear()

# --- Helper: Convert row, col to shogi.SQUARES index ---
def get_shogi_square(col, row):  # 0-indexed
    file = 9 - col  # shogi file from 9 to 1 (left to right)
    rank = row + 1  # shogi rank from a to i (top to bottom)
    square_str = f"{file}{chr(ord('a') + row)}"
    return shogi.SQUARES.get(square_str, None)

# --- Place Detected Pieces on the Shogi Board ---
for row in range(9):
    for col in range(9):
        piece = grid[row][col]
        if piece:
            try:
                square = get_shogi_square(col, row)
                # Convert 'P' to shogi.PAWN (1), lowercase = black, uppercase = white
                symbol = piece.lower().replace("+", "")
                promote = piece.startswith("+")

                # Map piece symbol to numeric constant
                piece_code = shogi.PIECE_SYMBOLS.index(symbol)
                shogi_piece = shogi.Piece(shogi.BLACK, piece_code, promote=promote)

                if square is not None:
                    board.set_piece_at(square, shogi_piece)
            except Exception as e:
                print(f"Error placing {piece} at {row},{col}: {e}")

# --- Print Final Board Representation ---
print("\n♟️ Final Detected Shogi Board:")
print(board.kif_str())

# Optional: Save visual output
# for visualizing boxes on image:
annotated = results.plot()
cv2.imwrite("output_annotated.jpg", annotated)
