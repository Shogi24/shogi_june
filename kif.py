from ultralytics import YOLO
import cv2
import shogi

# --- Load YOLOv8 Model ---
model = YOLO("polyk (b2-100).pt")  # Replace with your actual trained model path

# --- Load Image ---
image_path = "sample.jpg"
img = cv2.imread(image_path)
height, width = img.shape[:2]

# --- Run Detection ---
results = model(img)[0]

# --- Define 9x9 Grid ---
cell_w = width / 9
cell_h = height / 9
grid = [["" for _ in range(9)] for _ in range(9)]

# --- Map YOLO detections to grid ---
for box in results.boxes:
    label = results.names[int(box.cls[0].item())]

    x1, y1, x2, y2 = box.xyxy[0]
    x_center = float((x1 + x2) / 2)
    y_center = float((y1 + y2) / 2)

    col = int(x_center // cell_w)
    row = int(y_center // cell_h)

    if 0 <= row < 9 and 0 <= col < 9:
        grid[row][col] = label

# --- Clean KIF-style Print with Column Headers ---
print("\n📋 Detected Shogi Board (KIF-Style with Your Class Labels):\n")

# Column header (1 to 9)
col_labels = "  ".join([f"{i+1:^10}" for i in range(9)])
print(f"{'':<3}{col_labels}")

# Print each row with row label (a to i)
for row_idx, row in enumerate(grid):
    row_str = ""
    for col in range(9):
        cell = row[col].strip() if row[col] else "."
        row_str += f"{cell:^10}"
    print(f"{chr(ord('a') + row_idx)} | {row_str}")

# --- Save Annotated Image with Labels ---
annotated_img = results.plot()
cv2.imwrite("output_annotated.jpg", annotated_img)
print("\n🖼️ Annotated image saved as 'output_annotated.jpg'")
