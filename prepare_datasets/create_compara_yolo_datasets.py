import shutil
from pathlib import Path

# ================= CONFIG ==============
source_dataset = 'AutoFish'
second_dataset = 'AZTI'
split_base = Path(f"comparative_splits/{source_dataset}_SOURCE/{source_dataset}_{second_dataset}")
dataset_base = Path("datasets")

to_replace_folder = split_base / f"{source_dataset}_to_be_replaced"
replacement_folder = split_base / f"{second_dataset}_replacements"

#second_dataset = second_dataset.lower()
source_dataset= 'autofish'
second_dataset = 'catchhawk'
base_train_txt = "0.txt"   # 🔴 key change
all_replaced_train_txt = '100.txt'
to_replace_txts = ["25.txt", "50.txt", "75.txt"]
replacement_txts = ["25.txt", "50.txt", "75.txt"]

dataset_old = dataset_base / source_dataset
dataset_new = dataset_base / second_dataset
output_datasets = [f"{source_dataset.lower()}_{second_dataset}_25%", f"{source_dataset.lower()}_{second_dataset}_50%", f"{source_dataset.lower()}_{second_dataset}_75%"]
# ==========================================


def read_list(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

from pathlib import Path

IMG_EXT_CANDIDATES = [".jpg", ".jpeg", ".png", ".bmp"]


def resolve_image_path(base_path, split, filename):
    """
    Try to find image with given filename.
    If not found, try alternative extensions.
    """
    img_path = base_path / f"images/{split}" / filename

    if img_path.exists():
        return img_path

    # fallback: try other extensions
    stem = Path(filename).stem

    for ext in IMG_EXT_CANDIDATES:
        alt_path = base_path / f"images/{split}" / (stem + ext)
        if alt_path.exists():
            return alt_path

    return None

def copy_pair(src_dataset, filename, dst_dataset, split, new_filename=None):
    if new_filename is None:
        new_filename = filename

    src_img = resolve_image_path(src_dataset, split, filename)
    src_lbl = src_dataset / f"labels/{split}" / (Path(filename).stem + ".txt")

    if not src_img.exists() or not src_lbl.exists():
        print(f"[WARN] Missing: {filename} ({split})")
        return False

    dst_img = dst_dataset / f"images/{split}" / src_img.name
    dst_lbl = dst_dataset / f"labels/{split}" / (Path(new_filename).stem + ".txt")

    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_lbl, dst_lbl)

    return True


def copy_val_split(src_dataset, dst_dataset):
    print("Step 3: Copy VAL split (from replacement dataset)")

    for img_file in (src_dataset / "images/val").glob("*.*"):
        copy_pair(src_dataset, img_file.name, dst_dataset, "val")


def build_dataset(to_replace_txt, replacement_txt, output_name):
    print(f"\n==============================")
    print(f"Building dataset: {output_name}")
    print(f"==============================")

    output_dataset = dataset_base / output_name

    base_list = read_list(to_replace_folder / base_train_txt)  # 🔴 base set
    to_replace_list = read_list(to_replace_folder / to_replace_txt)
    replacement_list = read_list(replacement_folder / replacement_txt)

    assert len(to_replace_list) == len(replacement_list)

    replace_map = dict(zip(to_replace_list, replacement_list))

    replaced_count = 0
    unchanged =0
    b = 0
    print("Step 1 & 2: Build TRAIN from 0% base")
    print('replace_map', len(replace_map))
    for filename in base_list:
        if filename in replace_map:
            b+=1
            # replace from new dataset
            success = copy_pair(
                dataset_new,
                replace_map[filename],
                output_dataset,
                "train",
                new_filename=None
            )
            if success:
                replaced_count += 1
        else:
            unchanged +=1
            # keep original
            copy_pair(dataset_old, filename, output_dataset, "train")
    for filename in replace_map:
        if filename not in base_list:
            print(filename)
    print(f"[INFO]to be replaced {b} Replaced {replaced_count}/{len(base_list)} Unchanged: {unchanged}"
          f"({replaced_count / len(base_list):.2%})")

    # -------- VAL --------
    copy_val_split(dataset_new, output_dataset)

    print(f"✔ Done: {output_name}")

def build_train_only_dataset(output_name):
    print(f"\n==============================")
    print(f"Building dataset: {output_name} (TRAIN ONLY)")
    print(f"==============================")

    output_dataset = dataset_base / output_name

    base_list = read_list(to_replace_folder / base_train_txt)

    print("Step 1: Copy TRAIN from 0% base (no replacement)")

    for filename in base_list:
        copy_pair(dataset_old, filename, output_dataset, "train")

    print(f"[INFO] Total images: {len(base_list)}")

    # Optional: keep val split consistent with others
    copy_val_split(dataset_old, output_dataset)

    print(f"✔ Done: {output_name}")

def build_train_only_dataset_new(output_name):
    print(f"\n==============================")
    print(f"Building dataset: {output_name} (TRAIN ONLY)")
    print(f"==============================")

    output_dataset = dataset_base / output_name

    base_list = read_list(replacement_folder / all_replaced_train_txt)

    print("Step 1: Copy TRAIN from 100% replacement)")

    for filename in base_list:
        copy_pair(dataset_new, filename, output_dataset, "train")

def main():
    #build_train_only_dataset(f"{source_dataset.lower()}_{second_dataset}_0%")
    build_train_only_dataset_new(f"{source_dataset.lower()}_{second_dataset}_100%")

    #for t_txt, r_txt, out_name in zip(
    #    to_replace_txts,
    #    replacement_txts,
    #    output_datasets
    #):
    #    build_dataset(t_txt, r_txt, out_name)


if __name__ == "__main__":
    main()
