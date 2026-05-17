import os
import torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# ---------------- CONFIG ----------------
source = 'catchmonitor'
target = 'autofish'
ratio = 100
DATASET_NAME = f'coco_{source}_{target}_{ratio}%'
DATASET_NAME_PRETRAINED = f'coco_{source}_{target}_0%'
DATASET_ROOT = f"datasets/{DATASET_NAME}"  # your dataset folder
OUTPUT_DIR = f"./coco_results/{DATASET_NAME}"
IMG_SIZE = 640

SPLITS = {
    "train": "instances_train.json",
    "val": "instances_val.json",
}
# ----------------------------------------

# Register datasets
for split, ann_file in SPLITS.items():
    img_folder = os.path.join(DATASET_ROOT, "images", split)
    register_coco_instances(
        f"fish_{split}",
        {},
        os.path.join(DATASET_ROOT, "annotations", ann_file),
        img_folder
    )

# ---------------- CONFIGURATION ----------------
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.DATASETS.TRAIN = ("fish_train",)
cfg.DATASETS.TEST = ("fish_val",)
cfg.DATALOADER.NUM_WORKERS = 4

#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
#    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
#)
cfg.MODEL.WEIGHTS = f"coco_results/{DATASET_NAME_PRETRAINED}/model_best.pth"  # path to your trained model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # single class fish
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0001
num_train_images = len(DatasetCatalog.get("fish_train"))
batch_size = cfg.SOLVER.IMS_PER_BATCH
epochs = 200
cfg.SOLVER.STEPS = []
cfg.SOLVER.MAX_ITER = int((num_train_images / batch_size) * epochs)
print(f"Training for {epochs} epochs -> MAX_ITER={cfg.SOLVER.MAX_ITER}")

cfg.INPUT.MIN_SIZE_TRAIN = IMG_SIZE
cfg.INPUT.MAX_SIZE_TRAIN = IMG_SIZE
cfg.INPUT.MIN_SIZE_TEST = IMG_SIZE
cfg.INPUT.MAX_SIZE_TEST = IMG_SIZE
cfg.INPUT.CROP.ENABLED = True
cfg.INPUT.CROP.TYPE = "absolute"
cfg.INPUT.CROP.SIZE = [IMG_SIZE, IMG_SIZE]

cfg.TEST.EVAL_PERIOD = 500  # run evaluation every 500 iterations

cfg.OUTPUT_DIR = OUTPUT_DIR
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# ---------------- CUSTOM TRAINER ----------------
class BestModelTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.best_score = 0.0
        self.best_model_path = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
        self.evaluator = COCOEvaluator("fish_val", cfg, False, output_dir=cfg.OUTPUT_DIR)

    def test_and_save_results(self):
        val_loader = build_detection_test_loader(self.cfg, "fish_val")
        results = inference_on_dataset(self.model, val_loader, self.evaluator)
        # Use segmentation AP for Mask R-CNN
        score = results["segm"]["AP"] if "segm" in results else results["bbox"]["AP"]
        print(f"Validation AP: {score:.4f}")
        if score > self.best_score:
            self.best_score = score
            torch.save(self.model.state_dict(), self.best_model_path)
            print(f"Saved new best model with AP={score:.4f}")
        return results

    # Hook to override periodic test
    def build_hooks(self):
        hooks = super().build_hooks()
        # Remove default EvalHook
        hooks = [h for h in hooks if h.__class__.__name__ != "EvalHook"]
        # Add custom hook
        from detectron2.engine.hooks import HookBase
        class EvalHook(HookBase):
            def after_step(self):
                next_iter = self.trainer.iter + 1
                if next_iter % self.trainer.cfg.TEST.EVAL_PERIOD == 0:
                    self.trainer.test_and_save_results()
        hooks.append(EvalHook())
        return hooks

# ---------------- TRAIN ----------------
trainer = BestModelTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

