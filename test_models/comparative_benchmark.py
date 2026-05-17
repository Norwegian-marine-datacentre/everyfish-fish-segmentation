from ultralytics import YOLO
import numpy as np
from pathlib import Path
import cv2

# --------------------------
# Configuration
# --------------------------
training_dataset='catchmonitor'
testing_dataset='catchmonitor_autofish_combined_testsets'
CONF_THRESHOLD = 0.25
IMG_SIZE = 640
BBOX_MATCH_THRESHOLD = 0.5  # IoU threshold for matching

# --------------------------
# Utility functions
# --------------------------
def mask_iou(pred_mask, gt_mask):
    """Compute IoU between two binary masks."""
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return inter / union if union > 0 else 0.0
def mask_to_bbox(mask):
    """Compute bounding box (x1,y1,x2,y2) of a boolean mask"""
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None  # empty mask
    return [xs.min(), ys.min(), xs.max(), ys.max()]

def bbox_iou(boxA, boxB):
    """Compute IoU between two boxes"""
    if boxA is None or boxB is None:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0.0

def load_gt_masks_polygons(image_path, img_shape, GT_MASKS_DIR):
    """Load GT polygon masks as list of boolean masks"""
    img_h, img_w = img_shape
    masks = []
    txt_path = Path(GT_MASKS_DIR) / f"{Path(image_path).stem}.txt"
    if not txt_path.exists():
        return masks
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # at least 3 points
                continue
            coords = list(map(float, parts[1:]))
            pts = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * img_w)
                y = int(coords[i+1] * img_h)
                pts.append([x, y])
            pts = np.array([pts], dtype=np.int32)
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, pts, 1)
            masks.append(mask.astype(bool))
    return masks

def load_image_shape(image_path):
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    return (h, w)

def compute_iou(predictions, gt_masks_dir):
    instance_ious = []
    for r in predictions:
        if r.masks is None:
            continue
    
        pred_masks = r.masks.data.cpu().numpy() > 0.5  # predicted masks [N,H,W]
        h, w = pred_masks.shape[1], pred_masks.shape[2]
    
        gt_masks_orig = load_gt_masks_polygons(r.path, load_image_shape(r.path), gt_masks_dir)
        # resize GT masks to prediction size
        gt_masks = [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                    for m in gt_masks_orig]
    
        # Keep track of matched GTs
        matched_gt = set()
        for pm in pred_masks:
            pm_bbox = mask_to_bbox(pm)
            best_iou = 0.0
            best_idx = -1
            for i, gm in enumerate(gt_masks):
                if i in matched_gt:
                    continue
                gm_bbox = mask_to_bbox(gm)
                iou_bbox = bbox_iou(pm_bbox, gm_bbox)
                if iou_bbox > best_iou:
                    best_iou = iou_bbox
                    best_idx = i
            
            # Check bbox IoU threshold
            if best_iou >= BBOX_MATCH_THRESHOLD and best_idx >= 0:
                matched_gt.add(best_idx)
                # compute mask IoU for matched pair
                mask_iou_val = mask_iou(pm, gt_masks[best_idx])
                instance_ious.append(mask_iou_val)
            else:
                instance_ious.append(0.0)  # unmatched
    
    # Compute mean instance-level mIoU
    instance_miou = float(np.mean(instance_ious)) if instance_ious else 0.0
    return instance_miou
