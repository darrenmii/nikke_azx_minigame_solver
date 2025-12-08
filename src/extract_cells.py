import cv2
import numpy as np
import os
import shutil
from detect_grid import detect_grid

def extract_cells(image_path, output_dir="extracted_cells"):
    print(f"Extracting cells from {image_path}...")
    
    # Get Grid
    row_lines, col_lines = detect_grid(image_path, debug=False)
    if not row_lines or not col_lines:
        print("Failed to detect grid.")
        return

    # Prepare output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    cells_count = 0
    
    # Iterate through grid
    # row_lines define the horizontal boundaries (y coords)
    # col_lines define the vertical boundaries (x coords)
    
    for r in range(len(row_lines) - 1):
        for c in range(len(col_lines) - 1):
            y1 = row_lines[r]
            y2 = row_lines[r+1]
            x1 = col_lines[c]
            x2 = col_lines[c+1]
            
            # Crop
            # Ensure bounds
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)
            
            cell = img[y1:y2, x1:x2]
            
            if cell.size == 0: continue
            
            # Preprocessing
            # 1. Resize to constant size (e.g. 32x32) for ML
            # But let's keep original aspect first to see what we have
            
            # 2. Heuristic Cleaning (Optional)
            # User worried about background at top.
            # We can try to center the mass of the digit.
            
            # Save raw cell
            filename = f"cell_{r:02d}_{c:02d}.png"
            cv2.imwrite(os.path.join(output_dir, filename), cell)
            cells_count += 1
            
    print(f"Extracted {cells_count} cells to '{output_dir}'.")
    
    # Create a contact sheet (montage) for easy review
    create_montage(output_dir, cells_count, len(col_lines)-1)

def create_montage(input_dir, total_cells, cols):
    files = sorted(os.listdir(input_dir))
    if not files: return
    
    # Read first to get size
    sample = cv2.imread(os.path.join(input_dir, files[0]))
    h, w = sample.shape[:2]
    
    rows = (total_cells + cols - 1) // cols
    
    montage = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= total_cells: break
            
            f = files[idx]
            cell = cv2.imread(os.path.join(input_dir, f))
            # Resize if needed (should be same from grid)
            cell = cv2.resize(cell, (w, h))
            
            montage[r*h:(r+1)*h, c*w:(c+1)*w] = cell
            idx += 1
            
    output_path = f"montage_{os.path.basename(input_dir)}.png"
    cv2.imwrite(output_path, montage)
    print(f"Created montage: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        extract_cells(sys.argv[1])
    else:
        extract_cells('image01.png')
