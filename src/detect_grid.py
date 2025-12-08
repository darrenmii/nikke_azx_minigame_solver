import cv2
import numpy as np
import os

def detect_grid(image_path, debug=False):
    print(f"Processing {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Preprocessing to isolate numbers
    # The numbers are likely lighter/darker than background.
    # Adaptive threshold is usually good, but let's tune it to be less sensitive to background textures.
    # Tuned for image02: Block=61, C=20 seems robust.
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 61, 20)

    # cv2.imwrite(f"thresh_debug_{os.path.basename(image_path)}", thresh)

    # 2. Find Contours (Candidates for digits)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Filter Contours
    # We assume digits have some significant size.
    # User feedback: "Big blue boxes are numbers, small ones are noise."
    # Let's collect all candidate rects first, then filter by size statistics.
    
    candidates = []
    min_area_absolute = 10 # Very loose, just to remove single pixels
    
    candidate_area_max = h * w * 0.90
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        c_area = cw * ch
        if c_area > min_area_absolute and c_area < candidate_area_max:
            # Aspect ratio: Digits are usually tall or square.
            # Ratios like 0.2 (very thin line) or 5.0 (very wide line) are likely noise.
            ratio = ch / cw
            if 0.5 < ratio < 3.0: 
                candidates.append((x, y, cw, ch))

    if not candidates:
        print("No contours found after basic filtering.")
        return

    # Filter by Area - Keep only the "large" (digit) ones
    # Stats showed P75=234 but Median Digit Area=~1260. 
    # This implies digits are in the top quantile.
    areas = np.array([w*h for (x,y,w,h) in candidates])
    
    # Sort areas to find the size of the "big" objects
    sorted_areas = sorted(areas)
    n_candidates = len(sorted_areas)
    
    # Take the median of the top 20% largest contours as the "Digit Size" reference.
    # (Assuming digits are among the largest things, and there are a fair number of them)
    top_20_percent_idx = int(n_candidates * 0.8)
    large_areas = sorted_areas[top_20_percent_idx:]
    
    if not large_areas:
        print("Not enough candidates for stats.")
        return
        
    median_digit_area = np.median(large_areas)
    print(f"Median Area of Top 20% Contours: {median_digit_area}")
    
    # Threshold: Keep things that are at least 40% of the digit size
    area_threshold = median_digit_area * 0.4
    
    print(f"Dynamic Area Threshold: > {area_threshold:.1f}")
    
    digit_rects = []
    debug_img = img.copy()
    
    for (x, y, cw, ch) in candidates:
        area = cw * ch
        if area > area_threshold:
            digit_rects.append((x, y, cw, ch))
            # Draw candidates in Blue
            cv2.rectangle(debug_img, (x, y), (x+cw, y+ch), (255, 0, 0), 2)
        else:
            # Noise (do not draw or draw faint)
            pass

    print(f"Found {len(candidates)} candidates, kept {len(digit_rects)} large digit contours.")
    
    print(f"Area Threshold for Digits: > {area_threshold:.1f} pixels (Max area: {np.max(areas)})")
    
    for (x, y, cw, ch) in candidates:
        area = cw * ch
        if area > area_threshold:
            digit_rects.append((x, y, cw, ch))
            # Draw candidates in Blue for debug
            cv2.rectangle(debug_img, (x, y), (x+cw, y+ch), (255, 0, 0), 1)
        else:
            # Draw noise in Yellow (optional, maybe don't draw to keep clean)
            pass

    print(f"Found {len(candidates)} candidates, kept {len(digit_rects)} large digit contours.")
    if len(digit_rects) == 0:
        print("No valid digits found after size filtering.")
        return

    # 4. Grid Estimation from Candidates
    # We need to find the "step" (cell size).
    # We can calculate the distance to the nearest neighbor for every point.
    
    centers = np.array([(r[0] + r[2]//2, r[1] + r[3]//2) for r in digit_rects])
    
    # Sort by Y to find rows
    # But straight sorting fails if they are slightly misaligned.
    # Detailed approach:
    # A. Project centers to Y axis and find clusters (Rows)
    # B. Project centers to X axis and find clusters (Cols)
    
    # Using a 1D clustering (or histogram) approach for Y (Rows)
    y_coords = centers[:, 1]
    
    # Simple recursive clustering
    def cluster_1d(coords, tolerance=10):
        coords = sorted(coords)
        if not coords: return []
        clusters = []
        current_cluster = [coords[0]]
        
        for c in coords[1:]:
            if c - current_cluster[-1] < tolerance: # If close to previous, add to cluster
                current_cluster.append(c)
            else:
                # Close current, start new
                clusters.append(np.mean(current_cluster))
                current_cluster = [c]
        clusters.append(np.mean(current_cluster))
        return clusters

    # Tolerance should be < half a cell height.
    # If we don't know cell height, we can estimate it from the rect heights.
    avg_h = np.median([r[3] for r in digit_rects])
    avg_w = np.median([r[2] for r in digit_rects])
    print(f"Median Digit Height: {avg_h}, Median Digit Width: {avg_w}")
    
    row_centers = cluster_1d(y_coords, tolerance=avg_h * 0.5)
    
    # Now valid X coordinates
    x_coords = centers[:, 0]
    col_centers = cluster_1d(x_coords, tolerance=avg_w * 0.5)
    
    print(f"Estimated {len(row_centers)} rows and {len(col_centers)} cols based on clustering.")
    
    # Calculate intervals
    if len(row_centers) > 1:
        row_diffs = np.diff(row_centers)
        avg_row_step = np.median(row_diffs)
    else:
        avg_row_step = avg_h * 1.5 # fallback

    if len(col_centers) > 1:
        col_diffs = np.diff(col_centers)
        avg_col_step = np.median(col_diffs)
    else:
        avg_col_step = avg_w * 1.5 # fallback

    print(f"Estimated Grid Step: Row={avg_row_step:.1f}, Col={avg_col_step:.1f}")

    # 5. Construct Grid Lines
    # We want lines *between* the numbers.
    # Use the cluster centers (which are number centers) and shift by half step.
    
    # Refine grid: Start from the first center - half step ??
    # Better: The lines should be at center[i] + (center[i+1]-center[i])/2
    
    final_rows = []
    # Add top border
    if len(row_centers) > 0:
        final_rows.append(int(row_centers[0] - avg_row_step/2))
        
        for i in range(len(row_centers)-1):
            mid = (row_centers[i] + row_centers[i+1]) / 2
            final_rows.append(int(mid))
            
        # Add bottom border
        final_rows.append(int(row_centers[-1] + avg_row_step/2))

    final_cols = []
    if len(col_centers) > 0:
        final_cols.append(int(col_centers[0] - avg_col_step/2))
        
        for i in range(len(col_centers)-1):
            mid = (col_centers[i] + col_centers[i+1]) / 2
            final_cols.append(int(mid))
            
        final_cols.append(int(col_centers[-1] + avg_col_step/2))

    # Visualize Grid
    if debug:
        for y in final_rows:
            cv2.line(debug_img, (0, y), (w, y), (0, 0, 255), 2)
            
        for x in final_cols:
            cv2.line(debug_img, (x, 0), (x, h), (0, 255, 0), 2)
            
        output_filename = f"grid_debug_{os.path.basename(image_path)}"
        cv2.imwrite(output_filename, debug_img)
        print(f"Saved debug image to {output_filename}")
        
    return final_rows, final_cols

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        detect_grid(sys.argv[1])
    else:
        detect_grid('image01.png')
