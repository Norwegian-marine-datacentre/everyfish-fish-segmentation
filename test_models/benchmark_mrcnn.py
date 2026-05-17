# ================================
# Fish Dataset Evaluation Script
# Reports Mask mAP and Instance mIoU
# ================================

import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.detection_utils import read_image
from pycocotools import mask as maskUtils

# -------------------
# 1. Register your test dataset (COCO format)
# -------------------
DATASET_NAME = 'coco_catchwam_catchhawk_0%'
DATASET_ROOT = f"datasets/{DATASET_NAME}"  # your dataset folder

DatasetCatalog.clear()  # optional, clears previous registration
from detectron2.data.datasets import register_coco_instances
register_coco_instances(
    "fish_test",
    {},
    f"{DATASET_ROOT}/annotations/instances_test.json",
    f"{DATASET_ROOT}/images/test"
)
fish_meta = MetadataCatalog.get("fish_test")
MetadataCatalog.get("fish_test").thing_classes = ['fish']
print("Thing classes:", fish_meta.thing_classes)

# -------------------
# 2. Setup Detectron2 Mask R-CNN predictor
# -------------------
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.WEIGHTS = f"coco_results/{DATASET_NAME}/model_best.pth"  # path to your trained model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(fish_meta.thing_classes)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cuda"

predictor = DefaultPredictor(cfg)

# -------------------
# 3. Compute Mask mAP (COCO evaluator)
# -------------------
def compute_mask_map(cfg, dataset_name):
    evaluator = COCOEvaluator(
        dataset_name,
        cfg,
        False,
        output_dir="./output"
    )
    test_loader = build_detection_test_loader(cfg, dataset_name)
    results = inference_on_dataset(predictor.model, test_loader, evaluator)
    return results["segm"]  # dictionary with AP, AP50, AP75, etc.

segm_results = compute_mask_map(cfg, "fish_test")
print("\nMask mAP results:")
for k, v in segm_results.items():
    print(f"{k}: {v:.3f}")

# -------------------
# 4. Mask IoU utility
# -------------------
def mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return 0.0 if union == 0 else intersection / union

# -------------------
# 5. Compute Instance mIoU
# -------------------
def compute_instance_miou(dataset_name, predictor, iou_thresh=0.5):
    dataset = DatasetCatalog.get(dataset_name)

    total_iou = 0.0
    total_matches = 0
    total_fp = 0
    total_fn = 0

    for data in dataset:
        # read image
        image = read_image(data["file_name"], format="BGR")
        outputs = predictor(image)

        # predicted masks
        instances = outputs["instances"].to("cpu")
        pred_masks = instances.pred_masks.numpy() if len(instances) > 0 else []

        # GT masks
        gt_masks = []
        for ann in data["annotations"]:
            if "segmentation" not in ann:
                continue
            rle = maskUtils.frPyObjects(
                ann["segmentation"],
                data["height"],
                data["width"]
            )
            m = maskUtils.decode(rle)
            if m.ndim == 3:
                m = np.any(m, axis=2)
            gt_masks.append(m.astype(bool))

        matched_gt = set()

        # Greedy one-to-one matching
        for pm in pred_masks:
            best_iou = 0.0
            best_idx = -1
            for i, gm in enumerate(gt_masks):
                if i in matched_gt:
                    continue
                iou = mask_iou(pm, gm)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= iou_thresh:
                matched_gt.add(best_idx)
                total_iou += best_iou
                total_matches += 1
            else:
                total_fp += 1  # unmatched prediction

        total_fn += (len(gt_masks) - len(matched_gt))  # missed GTs

    denom = total_matches + total_fp + total_fn
    return 0.0 if denom == 0 else total_iou / denom

instance_miou = compute_instance_miou("fish_test", predictor, iou_thresh=0.5)
print(f"\nInstance mIoU: {instance_miou:.4f}")

# -------------------
# 6. Done
# -------------------
print("\nEvaluation complete!")

