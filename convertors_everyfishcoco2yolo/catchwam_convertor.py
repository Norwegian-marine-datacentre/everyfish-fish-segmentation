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
        segment = [[float(point['x']) / w, float(point['y']) / h] for point in polygon]
        segments.append(segment)
    if len(segments) > 1:
        print('multiple segments') # in catchhawk there is no overlapping
        merged_segments = merge_multi_segment(segments)
        merged_segments = np.concatenate(merged_segments, axis=0).reshape(-1).tolist()

    else:
        merged_segments = [i for j in segments[0] for i in j]
    return merged_segments

ROOT = Path("catchwam")                      # main folder
root = ROOT / "labels"

OUT = Path("catchwam_yolo")

ann_paths = [
    root / "train",
    root / "val",
    root / "test"
]

out_root = OUT / "labels"
out_root.mkdir(exist_ok=True)

# ================== CONVERSION ===========================

for ann_dir in ann_paths:
    set_name = ann_dir.name       # 'train', 'val', 'test'
    out_dir = out_root / set_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for js in ann_dir.glob("*.json"):
        with open(js, "r") as f:
            data = json.load(f)

        w = data["image"]["width"]
        h = data["image"]["height"]
        yolo_lines = []

        for ann in data["annotation"]:
            cls = ann["category_id"]


            seg = ann["segmentation"]  # [[x1,y1,x2,y2,...]]
            seg_converted = []
            for s in seg:
                s_converted = [{"x": s[i], "y": s[i + 1]} for i in range(0, len(s), 2)]
                seg_converted.append(s_converted)

            poly = process_segmentation(seg_converted, float(h), float(w))
            # YOLO format → class x1 y1 x2 y2 ... xn yn
            yolo_line = str(cls) + " " + " ".join(f"{v:.6f}" for v in poly) +'\n'
            yolo_lines.append(yolo_line)

        # Write YOLO file with same name but .txt extension
        out_file = out_dir / js.with_suffix(".txt").name
        with open(out_file, "w") as out:
            for yolo_line in yolo_lines:
                out.write(yolo_line)

        print(f"✔ {js.name} → {out_file}")

print("\n🎉 Completed! YOLO labels saved under:", out_root)