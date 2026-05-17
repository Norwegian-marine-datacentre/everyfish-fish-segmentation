from ultralytics import YOLO
import yaml
import numpy as np
import json
from comparative_benchmark import compute_iou 
candidate_datasets =  ['uea', 'autofish', 'azti', 'catchwam']
ratios = ['0', '25', '50', '75', '100']
result_dict = {}
for source in candidate_datasets:
    result_dict[source] = {}
    for ratio in ratios:
        result_dict[source][ratio] = {}
        for target in candidate_datasets:
            if source == target:
                continue
            dataset = f'agnostic_{source}_{target}_{ratio}'
            model = YOLO(f"runs/segment/{dataset}/weights/best.pt")
            with open(f"{dataset}.yaml", 'r') as f:
                data = yaml.safe_load(f)
            re_line = []
            for test_split in ['source', 'target', 'combined']:
                results = model.val(
                        data=f"{dataset}.yaml",
                        split=f"test_{test_split}",
                        imgsz=640,
                        save=False,
                        plots=False
                        )
                
                predictions = model.predict(
                    source=data[f"test_{test_split}"],
                    imgsz=640,
                    conf=0.25,
                    save=False
                )
                gt_masks_dir =  data[f"test_{test_split}"].replace('images', 'labels')
                mean_iou = compute_iou(predictions, gt_masks_dir)
                re_line.append(f'{mean_iou:0.3f} | {results.seg.map50:0.3f} | {results.seg.map:0.3f}')
            result_dict[source][ratio][target] = re_line
for source in result_dict:
    with open(f'yolo_result_{source}.json', 'w') as f:
        json.dump(result_dict[source], f, indent=4)
