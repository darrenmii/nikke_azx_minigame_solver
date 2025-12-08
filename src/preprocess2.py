import cv2
import numpy as np
import os
import sys
import glob

def process_image_to_normalized_digit(image_input, canvas_size=28, target_digit_size=20):
    """
    完整圖像處理流程。
    讀取一張圖片，經過所有處理步驟，返回一個 28x28 的標準化 numpy array。

    :param image_input: 輸入圖片的路徑 (str) 或 已經讀取的圖片 (numpy array)。
    :param canvas_size: 最終畫布的大小 (預設 28)。
    :param target_digit_size: 畫布內數字的目標大小 (預設 20)。
    :return: 處理完成的 28x28 numpy array，如果過程中出錯則返回 None。
    """
    # =================================================================
    # 步驟 1: 讀取圖片，轉為灰階，閾值化，反轉顏色
    # =================================================================
    if isinstance(image_input, str):
        original_img = cv2.imread(image_input)
        if original_img is None:
            print(f"警告：無法讀取圖片 {image_input}，跳過。")
            return None
    elif isinstance(image_input, np.ndarray):
        original_img = image_input
    else:
        print("錯誤: image_input 必須是檔案路徑或 numpy array")
        return None

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # 閾值化：將數字變為白色，背景變為黑色
    # 注意：這裡的 THRESH_BINARY_INV 可能與原腳本的兩步驟操作(THRESH_BINARY + bitwise_not)等效
    # 但為保持邏輯一致，我們遵循原腳本的步驟
    _ , binary_img = cv2.threshold(gray_img, 210, 255, cv2.THRESH_BINARY)
    # 反轉：數字變為黑色，背景變為白色
    inverted_img = cv2.bitwise_not(binary_img)

    # =================================================================
    # 步驟 2: 尋找內部輪廓，提取實心數字
    # =================================================================
    contours, hierarchy = cv2.findContours(inverted_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    solid_digit_mask = np.zeros_like(inverted_img)

    if hierarchy is not None:
        for i, h in enumerate(hierarchy[0]):
            parent_idx = h[3]
            # 條件：必須有父輪廓 (代表是孔洞) 且面積大於一定值以過濾雜訊
            if parent_idx != -1 and cv2.contourArea(contours[i]) > 20:
                cv2.drawContours(solid_digit_mask, [contours[i]], -1, 255, thickness=cv2.FILLED)

    # =================================================================
    # 步驟 3: 結合實心輪廓與原始輪廓，恢復數字的孔洞
    # =================================================================
    # 為了進行 AND 運算，我們需要一個黑底白字（帶孔）的圖像
    # inverted_img 是白底黑字，所以我們需要再次反轉它
    raw_inverted_for_and = cv2.bitwise_not(inverted_img)
    # 交集運算，恢復孔洞
    perfect_digit = cv2.bitwise_and(solid_digit_mask, raw_inverted_for_and)

    # =================================================================
    # 步驟 4: 裁切、縮放並置中到標準畫布上
    # =================================================================
    coords = cv2.findNonZero(perfect_digit)
    # 如果 perfect_digit 是全黑的（例如，無法識別的儲存格），coords 會是 None
    if coords is None:
        # 返回一個空的 28x28 圖像
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    digit_roi = perfect_digit[y:y+h, x:x+w]

    # 等比縮放
    ratio = target_digit_size / max(w, h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    
    # 防止 new_w 或 new_h 為 0
    if new_w == 0 or new_h == 0:
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    resized_digit = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 貼到畫布中央
    final_tensor = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    offset_x = (canvas_size - new_w) // 2
    offset_y = (canvas_size - new_h) // 2
    final_tensor[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized_digit

    return final_tensor

def main():
    """
    主函式：接收輸入目錄和輸出目錄，處理圖片並儲存結果。
    """
    if len(sys.argv) < 3:
        print("用法: python test5.py <輸入目錄> <輸出目錄>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    if not os.path.isdir(input_directory):
        print(f"錯誤: 輸入目錄 '{input_directory}' 不是一個有效的目錄。")
        sys.exit(1)

    # 如果輸出目錄不存在，則建立它
    os.makedirs(output_directory, exist_ok=True)
    print(f"輸出將儲存至: '{output_directory}'")

    # 尋找所有常見格式的圖片檔案
    image_paths = glob.glob(os.path.join(input_directory, '*.png'))
    image_paths += glob.glob(os.path.join(input_directory, '*.jpg'))
    image_paths += glob.glob(os.path.join(input_directory, '*.bmp'))
    
    if not image_paths:
        print(f"在 '{input_directory}' 中找不到任何圖片檔案。")
        sys.exit(0)

    print(f"在 '{input_directory}' 中找到 {len(image_paths)} 張圖片，開始處理...")

    processed_count = 0
    for i, path in enumerate(image_paths):
        filename = os.path.basename(path)
        print(f"({i+1}/{len(image_paths)}) 正在處理: {filename}", end='')
        
        # 執行處理流程
        final_result_array = process_image_to_normalized_digit(path)
        
        if final_result_array is not None:
            # 建立輸出路徑並儲存檔案
            output_path = os.path.join(output_directory, filename)
            cv2.imwrite(output_path, final_result_array)
            print(f" -> 已儲存至 {output_path}")
            processed_count += 1
        else:
            print(f" -> 處理失敗，已跳過。")
    
    print(f"\n所有圖片處理完畢。共 {processed_count} 張圖片成功處理並儲存。")

if __name__ == '__main__':
    main()