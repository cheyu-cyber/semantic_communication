import numpy as np
import cv2

def process_panoptic_result(panoptic_id_map, segments_info):
    """
    panoptic_id_map: 2D array where pixel val = object ID
    segments_info: List of mappings {id: 1, category_id: 3, ...} provided by model
    """
    final_output = []

    # Get all unique object IDs in the image
    present_ids = np.unique(panoptic_id_map)

    for obj_id in present_ids:
        if obj_id == 0: continue # Skip void/black areas

        # 1. Create a binary mask for JUST this object
        mask = (panoptic_id_map == obj_id).astype(np.uint8) * 255

        # 2. Find Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 3. Get Bounding Box
        x, y, w, h = cv2.boundingRect(contours[0])

        # 4. Get Name (Lookup logic depends on your specific model)
        # category_id = segments_info[obj_id]['category_id']
        # name = CLASS_NAMES[category_id]
        name = "Car" # Placeholder

        final_output.append({
            "name": name,
            "box": [x, y, w, h],
            "contour": contours[0].tolist() # Convert to list for JSON/Printing
        })

    return final_output


def merge_yolo_and_semantic(semantic_mask, yolo_results):
    """
    semantic_mask: 2D array (H, W) from EfficientViT (values 0-18)
    yolo_results: The 'results' object from Ultralytics
    """
    # 1. Create a canvas starting with the Semantic (Background) info
    # We copy it so we don't destroy the original data
    panoptic_map = semantic_mask.copy()
    
    # 2. Loop through YOLO detections and OVERWRITE the background
    # YOLO results usually contain .masks.data (bitmaps)
    if yolo_results[0].masks is not None:
        masks = yolo_results[0].masks.data.cpu().numpy()  # Get all masks
        classes = yolo_results[0].boxes.cls.cpu().numpy() # Get class IDs
        
        for i, mask in enumerate(masks):
            # Resize YOLO mask to match Semantic map size if needed
            # (YOLO masks are often smaller, e.g., 640x640 vs 1024x2048)
            mask_resized = cv2.resize(mask, (panoptic_map.shape[1], panoptic_map.shape[0]))
            
            # Threshold: YOLO masks are floats (0.0 to 1.0), make them binary
            binary_mask = mask_resized > 0.5
            
            # 3. Assign a unique ID for this instance
            # We offset instance IDs (e.g., start at 1000) to avoid clashing with class IDs (0-18)
            instance_id = 1000 + i 
            
            # Overwrite the semantic pixels with this specific instance ID
            # "Where the YOLO mask is True, write the Instance ID"
            panoptic_map[binary_mask] = instance_id

    return panoptic_map

# --- How to use the output ---
# Pixel value 0   -> Road (from EfficientViT)
# Pixel value 10  -> Sky (from EfficientViT)
# Pixel value 1001 -> Car Instance #1 (from YOLO)
# Pixel value 1002 -> Car Instance #2 (from YOLO)