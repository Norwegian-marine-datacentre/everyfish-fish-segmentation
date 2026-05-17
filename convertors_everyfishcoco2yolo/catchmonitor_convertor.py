import json
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np

def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance.

    Args:
        arr1: (N, 2).
        arr2: (M, 2).

    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """
    Merge multi segments to one list. Find the coordinates with min distance between each segment, then connect these
    coordinates with one thin line to merge all segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]
                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]: idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def process_segmentation(regions, h, w):
    segments = []
    for polygon in regions:
        segment = [[point['x'] / w, point['y'] / h] for point in polygon]
        segments.append(segment)

    if len(segments) > 1:
        merged_segments = merge_multi_segment(segments)
        merged_segments = np.concatenate(merged_segments, axis=0).reshape(-1).tolist()

    else:
        try:
            merged_segments = [i for j in segments[0] for i in j]
        except:
            return None
    return merged_segments

ROOT = Path("catchmonitor2.0")                      # main folder
IMG_DIR = ROOT / "images"                   # contains all images
ANN_DIR = ROOT / "annotations"  # COCO annotation file
SPLITS = {                                  # text file → target folder
    "train": ROOT / 'splits' /"train.json",
    "val": ROOT / 'splits' / "val.json",
    "test": ROOT / 'splits' / "test.json"
}
OUT = Path("catchmonitor_yolo2.0")

for split in SPLITS.keys():
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)


# ================= PROCESS EACH SPLIT ==================
for split, json_file in SPLITS.items():
    # Load JSON list of images
    filenames = json.load(open(json_file))

    # Force .png → .jpg uniformly
    filenames = [str(Path(f)) for f in filenames]

    for file in tqdm(filenames):
        img_path = IMG_DIR / file
        json_path = ANN_DIR / (Path(file).stem + ".json")

        # --- Copy image ---
        shutil.copy(img_path, OUT / "images" / split / file)

        # --- Convert annotation to YOLO ---
        label_out = OUT / "labels" / split / f"{Path(file).stem}.txt"

        # if not json_path.exists():
        #     # No annotation → create an empty .txt (YOLO expects it)
        #     open(label_out, "w").close()
        #     continue

        with open(json_path) as jf:
            ann = json.load(jf)
        
        W, H = ann["image"]["width"], ann["image"]["height"]
        
        with open(label_out, "w") as out:
            for obj in ann.get("annotation", []):
                cls = obj["category_id"]          # assumes 0-based
                seg = obj["segmentation"]  # list of points
                poly = process_segmentation(seg, H, W)
                if poly == None or len(poly) == 4:
                    print(file)
                    continue
                out.write(str(cls) + " " + " ".join(f"{v:.6f}" for v in poly) + "\n")


print("\n🚀 Conversion complete!")
print(f"YOLO dataset saved to: {OUT.resolve()}")
