import cv2
import os
import argparse
import glob
from preprocess import preprocess_image
import uuid

def gather_data():
    parser = argparse.ArgumentParser(description='Extract potential digits from images for manual labeling.')
    parser.add_argument('input_pattern', type=str, help='Glob pattern for input images (e.g., "raw_images/*.png")')
    parser.add_argument('--output_dir', type=str, default='data/to_label', help='Directory to save extracted digits')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    image_paths = glob.glob(args.input_pattern)
    if not image_paths:
        print(f"No images found matching pattern: {args.input_pattern}")
        return

    print(f"Found {len(image_paths)} images. Processing...")

    count = 0
    for img_path in image_paths:
        try:
            print(f"Processing {img_path}...")
            # preprocess_image returns (cells, num_rows, num_cols)
            # cells is a list of dicts: {'row': r, 'col': c, 'image': np_array}
            result = preprocess_image(img_path)
            if not result:
                continue
            
            cells_data, _, _ = result
            
            for cell in cells_data:
                img_data = cell['image']
                # Generate a unique filename
                filename = f"extract_{uuid.uuid4().hex[:8]}.png"
                save_path = os.path.join(args.output_dir, filename)
                cv2.imwrite(save_path, img_data)
                count += 1
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Done. Extracted {count} images to {args.output_dir}")
    print("Please manually sort these images into folders data/custom_dataset/train/0, .../1, etc.")

if __name__ == "__main__":
    gather_data()
