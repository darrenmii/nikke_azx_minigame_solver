
import sys
import os
import torch
import numpy as np
import argparse
from torchvision import transforms
from model import DigitNet
from preprocess import preprocess_image
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='Recognize numbers in a grid image.')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: File {args.image_path} does not exist.")
        return

    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DigitNet().to(device)
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pth')
    if not os.path.exists(model_path):
        # Fallback to current dir if running from src or root specially
        model_path = 'model.pth'
        
    if not os.path.exists(model_path):
        print("Error: model.pth not found. Please run train.py first.")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    model.eval()

    # Preprocess
    print(f"Processing {args.image_path}...")
    try:
        cells_data, num_rows, num_cols = preprocess_image(args.image_path)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return

    if not cells_data:
        print("No digits found.")
        return

    print(f"Detected grid: {num_rows} rows x ~{num_cols} columns")

    # Prepare batch
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Construct the matrix
    # Initialize with None or -1
    matrix = [[-1 for _ in range(num_cols)] for _ in range(num_rows)]

    # We need to map row index (0..N) to the matrix row.
    # The preprocess logic returns row indices 0..N.
    # However, num_cols might be the max cols found in any row.
    # Let's fill what we have.

    # Collect all cell images
    cell_images = []
    cell_indices = []

    for cell in cells_data:
        # cell['image'] is numpy array (28,28) uint8
        # Convert to PIL for transform
        pil_img = Image.fromarray(cell['image'])
        tensor_img = transform(pil_img)
        cell_images.append(tensor_img)
        cell_indices.append((cell['row'], cell['col']))

    if not cell_images:
        print("No valid cells extracted.")
        return

    batch = torch.stack(cell_images).to(device) # [N, 1, 28, 28]

    # Inference
    with torch.no_grad():
        outputs = model(batch)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.cpu().numpy()

    # Fill matrix
    for idx, (r, c) in enumerate(cell_indices):
        if r < num_rows and c < num_cols:
            matrix[r][c] = predictions[idx]

    # Print Result
    print("\nRecognized Matrix:")
    print("-" * (num_cols * 4 + 1))
    for row in matrix:
        line = "| "
        for val in row:
            if val == -1:
                line += "   | " # Empty or missing
            else:
                line += f"{val}  | "
        print(line)
    print("-" * (num_cols * 4 + 1))

    # Also print as pure list of lists for easy copying
    print("\nPython List Format:")
    clean_matrix = [[int(val) if val != -1 else 0 for val in row] for row in matrix]
    print(clean_matrix)

if __name__ == "__main__":
    main()
