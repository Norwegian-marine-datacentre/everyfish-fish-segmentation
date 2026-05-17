import shutil
import yaml
from pathlib import Path

candidate_datasets =  ['uea', 'autofish', 'azti', 'catchwam']
ratios = ['0', '25', '50', '75', '100']
dataset_mapping = {
                    'uea': 'catchmonitor',
                    'autofish': 'autofish',
                    'azti': 'catchhawk',
                    'catchwam': 'catchwam'
                    }
for source in candidate_datasets:
    for target in candidate_datasets:
        if source == target:
            continue
        dst_img = Path(f'./datasets/{dataset_mapping[source]}_{dataset_mapping[target]}/images/test')
        dst_lbl = Path(f'./datasets/{dataset_mapping[source]}_{dataset_mapping[target]}/labels/test')
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        src_img = Path(f'./datasets/{dataset_mapping[source]}/images/test')
        src_lbl = Path(f'./datasets/{dataset_mapping[source]}/labels/test')
        shutil.copytree(src_img, dst_img, dirs_exist_ok=True)
        shutil.copytree(src_lbl, dst_lbl, dirs_exist_ok=True)
        src_img = Path(f'./datasets/{dataset_mapping[target]}/images/test')
        src_lbl = Path(f'./datasets/{dataset_mapping[target]}/labels/test')
        shutil.copytree(src_img, dst_img, dirs_exist_ok=True)
        shutil.copytree(src_lbl, dst_lbl, dirs_exist_ok=True)

        for ratio in ratios:
            yaml_path = Path(f'agnostic_{source}_{target}_{ratio}.yaml')
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            data['test_source'] = f'./datasets/{dataset_mapping[source]}/images/test'
            data['test_target'] = f'./datasets/{dataset_mapping[target]}/images/test'
            data['test_combined'] = f'./datasets/{dataset_mapping[source]}_{dataset_mapping[target]}/images/test'
            
            with open(yaml_path, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)

