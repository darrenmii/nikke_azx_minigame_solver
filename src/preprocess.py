import cv2
import numpy as np
import os
import sys

# Add src to path if needed (though usually relative imports work if package structure is correct)
# But here we are in same dir.
from detect_grid import detect_grid
from preprocess2 import process_image_to_normalized_digit

def preprocess_image(image_path):
    """
    Reads an image and extracts grid cells using detect_grid, 
    then processes each cell using preprocess2.
    
    Returns:
        cells_data: List of dicts {row, col, image, x, y, w, h}
        num_rows: Number of rows detected
        num_cols: Number of cols detected
    """
    print(f"Preprocessing {image_path} using new logic...")
    
    # 1. Detect Grid
    row_lines, col_lines = detect_grid(image_path, debug=False)
    
    if not row_lines or not col_lines:
        print("Failed to detect grid lines.")
        return [], 0, 0
        
    num_rows = len(row_lines) - 1
    num_cols = len(col_lines) - 1
    
    # 2. Extract and Process Cells
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    h_img, w_img = img.shape[:2]
    
    cells_data = []
    
    for r in range(num_rows):
        for c in range(num_cols):
            y1 = row_lines[r]
            y2 = row_lines[r+1]
            x1 = col_lines[c]
            x2 = col_lines[c+1]
            
            # Clamp to image bounds
            y1, y2 = max(0, y1), min(h_img, y2)
            x1, x2 = max(0, x1), min(w_img, x2)
            
            # Extract cell
            cell_img = img[y1:y2, x1:x2]
            
            if cell_img.size == 0:
                continue
                
            # Process cell to get 28x28 normalized digit
            # Note: process_image_to_normalized_digit now accepts numpy array
            processed_cell = process_image_to_normalized_digit(cell_img)
            
            if processed_cell is not None:
                cells_data.append({
                    'row': r,
                    'col': c,
                    'image': processed_cell, # 28x28 uint8
                    'x': x1,
                    'y': y1,
                    'w': x2 - x1,
                    'h': y2 - y1
                })
                
    print(f"Extracted {len(cells_data)} valid cells.")
    
    return cells_data, num_rows, num_cols

if __name__ == "__main__":
    # Test
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = 'image01.png'
        
    if os.path.exists(path):
        data, rows, cols = preprocess_image(path)
        print(f"Result: {rows}x{cols} grid, {len(data)} cells.")
    else:
        print(f"File {path} not found.")
