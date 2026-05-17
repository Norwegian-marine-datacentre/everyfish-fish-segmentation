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
source_dataset = "AutoFish"
second_dataset = "UEA"

dataset_base = Path("datasets")

split_base = Path(f"comparative_splits/{source_dataset}_SOURCE/{source_dataset}_{second_dataset}")

to_replace_folder = split_base / f"{source_dataset}_to_be_replaced"
replacement_folder = split_base / f"{second_dataset}_replacements"

base_train_txt = "0.txt"
all_replaced_train_txt = "100.txt"

to_replace_txts = ["25.txt", "50.txt", "75.txt"]
replacement_txts = ["25.txt", "50.txt", "75.txt"]
source_dataset = DATASET_MAPPING[source_dataset]
second_dataset = DATASET_MAPPING[second_dataset]
output_datasets = [
    f"coco_{source_dataset}_{second_dataset}_25%",
    f"coco_{source_dataset}_{second_dataset}_50%",
    f"coco_{source_dataset}_{second_dataset}_75%",
]

dataset_old = dataset_base / f'{source_dataset}_coco'
dataset_new = dataset_base / f'{second_dataset}_coco'
# ==========================================


# ---------- utils ----------
def read_list(path):
    with open(path, "r") as f:
        return [x.strip() for x in f if x.strip()]


def load_coco(path):
    with open(path, "r") as f:
        return json.load(f)


def save_coco(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


def build_index(coco):
    img_map = {img["file_name"]: img for img in coco["images"]}
    ann_map = {}
    for ann in coco["annotations"]:
        ann_map.setdefault(ann["image_id"], []).append(ann)
    return img_map, ann_map


# ---------- core builder ----------
def build_dataset_coco(base_list, replace_map, output_name, flag=True):
    print(f"\n==============================")
    print(f"Building: {output_name}")
    print(f"==============================")

    output_path = dataset_base / output_name
    (output_path / "images/val").mkdir(parents=True, exist_ok=True)
    (output_path / "images/train").mkdir(parents=True, exist_ok=True)
    (output_path / "annotations").mkdir(parents=True, exist_ok=True)

    # load coco
    base_coco = load_coco(dataset_old / "annotations/instances_train.json")
    new_coco = load_coco(dataset_new / "annotations/instances_train.json")

    base_imgs, base_anns = build_index(base_coco)
    new_imgs, new_anns = build_index(new_coco)

    output = {
        "images": [],
        "annotations": [],
        "categories": base_coco["categories"],  # same (1 category)
    }
    val_output = {
        "images": [],
        "annotations": [],
        "categories": base_coco["categories"],  # same (1 category)
    }

    global_img_id = 0
    global_ann_id = 0
    replaced = 0
    missing = 0

    for filename in base_list:
        # decide source
        if filename in replace_map:
            src_name = replace_map[filename]
            src_imgs, src_anns = new_imgs, new_anns
            src_root = dataset_new / "images/train"
            replaced += 1
        else:
            src_name = filename
            src_imgs, src_anns = base_imgs, base_anns
            src_root = dataset_old / "images/train"

        if src_name not in src_imgs:
            src_name = src_name.replace('png', 'jpg')
            if src_name not in src_imgs:
                print(f"[WARN] Missing: {src_name}")
                missing += 1
                continue

        img = src_imgs[src_name]
        anns = src_anns.get(img["id"], [])

        # new IDs
        global_img_id += 1
        new_img_id = global_img_id

        new_filename = filename  # preserve base naming

        # copy image
        src_path = src_root / img["file_name"]
        dst_path = output_path / "images/train" / img["file_name"]

        if not src_path.exists():
            src_path.with_suffix('.jpg')
            dst_path.with_suffix('.jpg')
        if not src_path.exists():
            print(f"[WARN] File missing: {src_path}")
            missing += 1
            continue

        shutil.copy2(src_path, dst_path)

        # add image
        output["images"].append({
            "id": new_img_id,
            "file_name": dst_path.name,
            "width": img["width"],
            "height": img["height"],
        })

        # add annotations
        for ann in anns:
            global_ann_id += 1
            output["annotations"].append({
                "id": global_ann_id,
                "image_id": new_img_id,
                "category_id": 0,
                "segmentation": ann['segmentation'],
                "bbox": ann["bbox"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": ann.get("iscrowd", 0),
            })

    print(f"[INFO] Replaced: {replaced}/{len(base_list)} | Missing: {missing}")

    # save train.json
    save_coco(output, output_path / "annotations/instances_train.json")

    # -------- VAL (copy from replacement dataset like your original) --------
    print("Copying VAL split...")

    val_src = dataset_new if flag else dataset_old
    val_coco = load_coco(val_src / "annotations/instances_val.json")
    val_imgs, val_anns = build_index(val_coco)
    print("Processing VAL...")

    for img in val_coco["images"]:

        src_path = val_src / "images/val" / img["file_name"]
        if not src_path.exists():
            src_path = src_path.with_suffix('.jpg')
        if not src_path.exists():
            print(f"[WARN] Missing val image: {img['file_name']}")
            continue

        global_img_id += 1
        new_img_id = global_img_id

        shutil.copy2(
            src_path,
            output_path / "images/val" / src_path.name
        )

        val_output["images"].append({
            "id": new_img_id,
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"]
        })

        for ann in val_anns.get(img["id"], []):
            global_ann_id += 1
            val_output["annotations"].append({
                "id": global_ann_id,
                "image_id": new_img_id,
                "category_id": 0,
                "segmentation": ann['segmentation'],
                "bbox": ann["bbox"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": ann.get("iscrowd", 0),
            })

    # ================= SAVE =================
    save_coco(val_output, output_path / "annotations/instances_val.json")


    print(f"✔ Done: {output_name}")


# ---------- wrappers ----------
def build_0_percent():
    base_list = read_list(to_replace_folder / base_train_txt)
    build_dataset_coco(base_list, {}, f"coco_{source_dataset}_{second_dataset}_0%", flag=False)


def build_100_percent():
    base_list = read_list(replacement_folder / all_replaced_train_txt)

    # identity map (all replaced)
    replace_map = {f: f for f in base_list}

    build_dataset_coco(base_list, replace_map, f"coco_{source_dataset}_{second_dataset}_100%")


def build_mixed():
    for t_txt, r_txt, out_name in zip(to_replace_txts, replacement_txts, output_datasets):
        base_list = read_list(to_replace_folder / base_train_txt)
        to_replace = read_list(to_replace_folder / t_txt)
        replacement = read_list(replacement_folder / r_txt)

        assert len(to_replace) == len(replacement)

        replace_map = dict(zip(to_replace, replacement))

        build_dataset_coco(base_list, replace_map, out_name)


# ---------- main ----------
def main():
    build_0_percent()
    build_100_percent()
    build_mixed()


if __name__ == "__main__":
    main()
