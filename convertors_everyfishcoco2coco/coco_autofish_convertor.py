import json
import os
import shutil

# -------- CONFIG --------
SPLITS = {
    "train": "instances_train.json",
    "val": "instances_val.json",
    "test": "instances_test.json"
}
INPUT_ROOT = "autofish/autofish"
IMAGE_ROOT = os.path.join(INPUT_ROOT, "images") # folder containing ALL original images
OUTPUT_ROOT = "autofish_coco"     # new structured dataset
# ------------------------

SINGLE_CATEGORY = [{
    "id": 0,
    "name": "fish",
    "supercategory": "fish"
}]

def process_split(split_name, json_file):
    json_path = os.path.join(INPUT_ROOT, json_file)

    with open(json_path, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    # force single class
    for ann in annotations:
        ann["category_id"] = 0

    # output dirs
    img_out = os.path.join(OUTPUT_ROOT, "images", split_name)
    ann_out = os.path.join(OUTPUT_ROOT, "annotations")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(ann_out, exist_ok=True)

    # copy images
    for img in images:
        src = os.path.join(IMAGE_ROOT, img["file_name"])
        dst = os.path.join(img_out, img["file_name"])

        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing image: {src}")

        shutil.copy2(src, dst)

    # save JSON
    new_coco = {
        "images": images,
        "annotations": annotations,
        "categories": SINGLE_CATEGORY
    }

    out_json = os.path.join(ann_out, f"instances_{split_name}.json")
    with open(out_json, "w") as f:
        json.dump(new_coco, f)

    print(f"{split_name}: {len(images)} images copied")

for split, jf in SPLITS.items():
    process_split(split, jf)

print("Conversion complete.")
