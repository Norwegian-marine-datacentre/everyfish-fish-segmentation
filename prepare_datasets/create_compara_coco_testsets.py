import json
import shutil
from pathlib import Path

# ================= CONFIG =================
DATASET_MAPPING = {
                    'AZTI': 'catchhawk',
                    'FDWE': 'catchwam',
                    'UEA': 'catchmonitor',
                    'AutoFish': 'autofish'
                    }
source_dataset = "UEA"
second_dataset = "AutoFish"

dataset_base = Path("datasets")

source_dataset = DATASET_MAPPING[source_dataset]
second_dataset = DATASET_MAPPING[second_dataset]
output_root = dataset_base / f"test_coco_{source_dataset}_{second_dataset}"

dataset_source = dataset_base / f'{source_dataset}_coco'
dataset_second = dataset_base / f'{second_dataset}_coco'
# ==========================================
# coco paths
source_images = dataset_source / "images" /'test' 
source_json = dataset_source / "annotations" / "instances_test.json"

target_images = dataset_second / "images" / 'test'
target_json = dataset_second / "annotations" / "instances_test.json"

# ==========================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f)


def copy_images(coco, src_dir, dst_dir):
    """
    Preserve original folder structure inside images/test
    """

    for img in coco["images"]:

        src = src_dir / img["file_name"]
        dst = dst_dir / img["file_name"]

        dst.parent.mkdir(parents=True, exist_ok=True)

        if not dst.exists():
            shutil.copy2(src, dst)


def remap_coco_ids(coco, image_start=0, ann_start=0):

    image_id_map = {}

    new_images = []
    new_annotations = []

    next_img_id = image_start
    next_ann_id = ann_start

    # remap image ids
    for img in coco["images"]:

        old_id = img["id"]

        new_img = img.copy()
        new_img["id"] = next_img_id

        image_id_map[old_id] = next_img_id

        new_images.append(new_img)

        next_img_id += 1

    # remap annotation ids
    for ann in coco["annotations"]:

        new_ann = ann.copy()

        new_ann["id"] = next_ann_id
        new_ann["image_id"] = image_id_map[ann["image_id"]]

        new_annotations.append(new_ann)

        next_ann_id += 1

    return {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco["categories"]
    }


def main():

    # ======================================
    # LOAD TEST DATASETS
    # ======================================

    source_coco = load_json(source_json)
    target_coco = load_json(target_json)

    # ======================================
    # REMAP IDS
    # ======================================

    source_coco = remap_coco_ids(
        source_coco,
        image_start=0,
        ann_start=0
    )

    target_coco = remap_coco_ids(
        target_coco,
        image_start=len(source_coco["images"]),
        ann_start=len(source_coco["annotations"])
    )

    # ======================================
    # SOURCE SPLIT
    # ======================================

    source_out = output_root / "source"

    copy_images(
        source_coco,
        source_images,
        source_out / "images" / "test"
    )

    save_json(
        source_coco,
        source_out / "annotations" / "instances_test.json"
    )

    # ======================================
    # TARGET SPLIT
    # ======================================

    target_out = output_root / "target"

    copy_images(
        target_coco,
        target_images,
        target_out / "images" / "test"
    )

    save_json(
        target_coco,
        target_out / "annotations" / "instances_test.json"
    )

    # ======================================
    # COMBINED SPLIT
    # ======================================

    combined_out = output_root / "combined"

    combined_coco = {
        "images": (
            source_coco["images"] +
            target_coco["images"]
        ),
        "annotations": (
            source_coco["annotations"] +
            target_coco["annotations"]
        ),
        "categories": source_coco["categories"]
    }

    # copy source images
    copy_images(
        source_coco,
        source_images,
        combined_out / "images" / "test"
    )

    # copy target images
    copy_images(
        target_coco,
        target_images,
        combined_out / "images" / "test"
    )

    save_json(
        combined_coco,
        combined_out / "annotations" / "instances_test.json"
    )

    print(f"Done: {output_root}")


if __name__ == "__main__":
    main()
