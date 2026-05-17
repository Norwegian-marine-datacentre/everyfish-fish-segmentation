import json
from pathlib import Path
from tqdm import tqdm

# ---------------- CONFIG ----------------
DATASET_ROOT = Path("catchmonitor2.0")
ANN_DIR = DATASET_ROOT / "annotations"
SPLIT_DIR = DATASET_ROOT / "splits"
OUTPUT_DIR = Path("catchmonitor_coco2.0/annotations")

SPLITS = ["train", "val", "test"]

CATEGORIES = [
    {"id": 0, "name": "fish", "supercategory": "fish"}
]
# --------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

global_image_id = 0
global_ann_id = 0


def convert_segmentation(seg):
    """
    Convert segmentation from:
    [[{"x": "...", "y": "..."}, ...]]
    to:
    [[x1, y1, x2, y2, ...]]
    """
    coco_segs = []

    for polygon in seg:
        if len(polygon) < 3:
            print(polygon)
            continue  # invalid polygon

        flat_poly = []
        for pt in polygon:
            flat_poly.append(float(pt["x"]))
            flat_poly.append(float(pt["y"]))

        if len(flat_poly) >= 6:
            coco_segs.append(flat_poly)

    return coco_segs


def polygon_to_bbox(segmentation):
    all_x = []
    all_y = []
    for poly in segmentation:
        xs = poly[0::2]  # even indices -> x
        ys = poly[1::2]  # odd indices -> y
        all_x.extend(xs)
        all_y.extend(ys)

    x_min = min(all_x)
    y_min = min(all_y)
    x_max = max(all_x)
    y_max = max(all_y)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def load_split_filenames(split):
    with open(SPLIT_DIR / f"{split}.json") as f:
        split_names = json.load(f)
        return split_names


def convert_split(split):
    global global_image_id, global_ann_id

    images = []
    annotations = []

    filenames = load_split_filenames(split)

    for filename in tqdm(filenames, desc=f"Converting {split}"):
        ann_path = ANN_DIR / f"{Path(filename).stem}.json"

        if not ann_path.exists():
            print(f"⚠ Missing annotation: {ann_path.name}")
            continue

        with open(ann_path) as f:
            data = json.load(f)

        # ---------- IMAGE ENTRY ----------
        images.append({
            "id": global_image_id,
            "file_name": filename,  # Detectron2 uses this later
            "width": int(data["image"]["width"]),
            "height": int(data["image"]["height"])
        })
        # ---------- ANNOTATIONS ----------
        for obj in data.get("annotation", []):
            if obj["category_id"] == 41:
                continue
            flat_seg = convert_segmentation(obj["segmentation"])
            if len(flat_seg) == 0:
                continue
            bbox = polygon_to_bbox(flat_seg)
            x, y, w, h = bbox

            annotations.append({
                "id": global_ann_id,
                "image_id": global_image_id,
                "category_id": 0,  # fish
                "bbox": [x, y, w, h],
                "area": w * h,
                "segmentation": flat_seg,
                "iscrowd": 0
            })

            global_ann_id += 1

        global_image_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES
    }

    out_file = OUTPUT_DIR / f"instances_{split}.json"
    with open(out_file, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"✅ {split}: {len(images)} images, {len(annotations)} annotations")


for split in SPLITS:
    convert_split(split)

print("🎉 COCO annotation conversion complete")
