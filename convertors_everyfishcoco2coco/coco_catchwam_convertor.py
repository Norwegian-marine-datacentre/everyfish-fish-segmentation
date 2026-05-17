import json
from pathlib import Path
from tqdm import tqdm

# ---------------- CONFIG ----------------
INPUT_ROOT = Path("catchwam/labels")  # folder containing train/val/test JSONs
OUTPUT_ROOT = Path("catchwam_coco/annotations")
SPLITS = ["train", "val", "test"]
SINGLE_CATEGORY = [{"id": 0, "name": "fish", "supercategory": "fish"}]

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
# ----------------------------------------

global_image_id = 0
global_ann_id = 0

def convert_split(split: str):
    """
    Convert per-image JSONs in a split folder to COCO format
    """
    global global_image_id, global_ann_id

    images = []
    annotations = []

    split_folder = INPUT_ROOT / split
    ann_files = list(split_folder.glob("*.json"))

    for ann_file in tqdm(ann_files, desc=f"Converting {split}"):
        with open(ann_file, "r") as f:
            data = json.load(f)

        # ---------------- Image entry ----------------
        image_info = data["image"]
        image_id = image_info["id"]
        width = image_info["width"]
        height = image_info["height"]
        file_name = image_info["file_name"]
        images.append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        # ---------------- Annotations ----------------
        for obj in data["annotation"]:
            ann = {
                "id": global_ann_id,
                "image_id": image_id,
                "category_id": 0,  # single class fish
                "iscrowd": 0
            }

            if "bbox" in obj:
                ann["bbox"] = obj["bbox"]
                x, y, w, h = obj["bbox"]
                ann["area"] = w * h
            if "segmentation" in obj:
                ann["segmentation"] = obj["segmentation"]

            annotations.append(ann)
            global_ann_id += 1


    # ---------------- Save COCO JSON ----------------
    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": SINGLE_CATEGORY
    }

    out_file = OUTPUT_ROOT / f"instances_{split}.json"
    with open(out_file, "w") as f:
        json.dump(coco_json, f, indent=4)
    print(f"{split}: {len(images)} images, {len(annotations)} annotations saved to {out_file}")


# ---------------- PROCESS ALL SPLITS ----------------
for split in SPLITS:
    convert_split(split)

print("All splits converted to COCO format successfully.")

