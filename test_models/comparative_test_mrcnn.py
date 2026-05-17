import yaml
import numpy as np
import json
from comparative_benchmark_mrcnn import compute_map_miou 
candidate_datasets =  ['autofish', 'catchhawk', 'catchwam', 'catchmonitor']
ratios = ['0', '25', '50', '75', '100']
result_dict = {}
for source in candidate_datasets:
    result_dict[source] = {}
    for ratio in ratios:
        result_dict[source][ratio] = {}
        for target in candidate_datasets:
            if source == target:
                continue
            dataset_name = f'coco_{source}_{target}_{ratio}%'
            dataset_root = f'test_coco_{source}_{target}'
            re_line = []
            for test_split in ['source', 'target', 'combined']:
                result = compute_map_miou(dataset_name, dataset_root, test_split)
                re_line.append(result)
            result_dict[source][ratio][target] = re_line
    with open(f'coco_result_{source}.json', 'w') as f:
        json.dump(result_dict[source], f, indent=4)
