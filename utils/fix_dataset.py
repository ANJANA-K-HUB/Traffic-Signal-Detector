import os
import cv2
import sys
import numpy as np

# Adjust these to point exactly to your local directories
IMAGES_DIR = "./dataset/train/images"
LABELS_DIR = "./dataset/train/labels"

def detect_dominant_color(img_patch):
    """Analyzes a cropped bounding box to see if it's predominantly Red or Green"""
    hsv = cv2.cvtColor(img_patch, cv2.COLOR_BGR2HSV)
    
    # Define HSV thresholds for Red and Green
    # Red wraps around 0 and 180 in OpenCV Hue channel
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    red_pixels = cv2.countNonZero(mask_red)
    green_pixels = cv2.countNonZero(mask_green)
    
    if green_pixels > red_pixels and green_pixels > 5:
        return 2  # Green Light Class ID
    elif red_pixels > green_pixels and red_pixels > 5:
        return 0  # Red Light Class ID
    return None

# Process all labels
fixed_count = 0

for label_file in os.listdir(LABELS_DIR):
    if not label_file.endswith('.txt'):
        continue
        
    label_path = os.path.join(LABELS_DIR, label_file)
    # Find matching image file (handling common extensions)
    base_name = os.path.splitext(label_file)[0]
    img_path = None
    for ext in ['.jpg', '.jpeg', '.png']:
        test_path = os.path.join(IMAGES_DIR, base_name + ext)
        if os.path.exists(test_path):
            img_path = test_path
            break
            
    if not img_path:
        continue
        
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    h, w, _ = img.shape
    new_lines = []
    modified = False
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
            
        cls_id = int(parts[0])
        
        # Only inspect boxes currently labeled as 0 (Red Light)
        if cls_id == 0 or cls_id == 1:
            # Convert normalized YOLO coordinates back to pixel coordinates
            x_center, y_center, box_w, box_h = map(float, parts[1:5])
            x1 = max(0, int((x_center - box_w/2) * w))
            y1 = max(0, int((y_center - box_h/2) * h))
            x2 = min(w, int((x_center + box_w/2) * w))
            y2 = min(h, int((y_center + box_h/2) * h))
            
            # Crop the bounding box patch
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                detected_cls = detect_dominant_color(crop)
                if detected_cls is not None and detected_cls != cls_id:
                    cls_id = detected_cls
                    modified = True
                    
        parts[0] = str(cls_id)
        new_lines.append(" ".join(parts) + "\n")
        
    if modified:
        with open(label_path, 'w') as f:
            f.writelines(new_lines)
        fixed_count += 1

print(file=sys.stderr)  # Flush output stream safely
print(f" Processed dataset successfully! Fixed {fixed_count} mislabeled annotation files.")