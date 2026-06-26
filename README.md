# Species-Agnostic Fish Segmentation

## Description

This repository contains the code for the EVERYFISH AI module **"Species-Agnostic Fish Segmentation"** — a comparative study that trains and benchmarks deep learning models for fish instance segmentation across multiple fishing vessel camera systems.

The study evaluates two state-of-the-art segmentation architectures:

- **YOLO11-Seg** (Ultralytics)
- **Mask R-CNN** (Detectron2, ResNet-50 FPN backbone)

Both models are trained and tested on four fish detection datasets collected from different commercial fishing vessels: **CatchMonitor**, **CatchHawk**, **CatchWam**, and **AutoFish**. The goal is to understand how well a model trained on images from one vessel/camera setup (the *source* domain) generalizes to images from another (the *target* domain), by mixing training data at different ratios (0%, 25%, 50%, 75%, 100% target-domain images).

All datasets are treated as **single-class** ("fish"), hence the term "species-agnostic" — the models segment fish regions without distinguishing between species.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [1. Download the Data](#1-download-the-data)
  - [2. Convert Data Formats](#2-convert-data-formats)
  - [3. Prepare Mixed Datasets](#3-prepare-mixed-datasets)
  - [4. Train Models](#4-train-models)
  - [5. Benchmark and Compare Models](#5-benchmark-and-compare-models)
- [License](#license)
- [Acknowledgement](#acknowledgement)
- [Contact](#contact)

---

## Installation

This project requires Python 3.8+ and a CUDA-capable GPU for training.

### 1. Clone the repository

```bash
git clone https://git.imr.no/everyfish/fish-segmentation.git
cd comparative_study_code
```

### 2. Create a virtual environment and install dependencies

```bash
# Create and activate a conda environment (recommended)
conda create -n fishseg python=3.10
conda activate fishseg

# Install PyTorch (choose the command matching your CUDA version from pytorch.org)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install core dependencies
pip install ultralytics        # YOLO11-Seg
pip install opencv-python tqdm pyyaml numpy

# Install Detectron2 (from pre-built wheel or source — see detectron2 documentation)
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# Install pycocotools
pip install pycocotools
```

---

## Usage

### 1. Download the Data

The raw datasets are provided in the **EVERYFISH-COCO** format. Each dataset contains images from a different fishing vessel camera setup, with per-image JSON annotation files that include polygon segmentation masks and bounding boxes for fish instances.
After downloading, extract each dataset into the project root. The expected directory structures are:

### 2. Convert Data Formats

The raw EVERYFISH-COCO format needs to be converted into formats expected by each framework:

- **COCO format** (for Mask R-CNN / Detectron2): standard COCO JSON with `images`, `annotations`, and `categories` keys
- **YOLO format** (for YOLO11-Seg): images in `images/{split}/` and `.txt` label files in `labels/{split}/`

Before running, edit the configuration block at the top of each script to point `INPUT_ROOT` / `DATASET_ROOT` to your raw data directory.

#### Convert to COCO format (Mask R-CNN)

```bash
python convertors_everyfishcoco2coco/coco_autofish_convertor.py
python convertors_everyfishcoco2coco/coco_catchhawk_convertor.py
python convertors_everyfishcoco2coco/coco_catchmonitor_convertor.py
python convertors_everyfishcoco2coco/coco_catchwam_convertor.py
```

Outputs go to `{dataset}_coco/` directories.

#### Convert to YOLO format

```bash
python convertors_everyfishcoco2yolo/autofish_convertor.py
python convertors_everyfishcoco2yolo/catchhawk_convertor.py
python convertors_everyfishcoco2yolo/catchmonitor_convertor.py
python convertors_everyfishcoco2yolo/catchwam_convertor.py
```

Outputs go to `{dataset}_yolo/` directories.

> **Note**: The YOLO converters merge multi-segment polygons into a single connected polygon per instance using a minimum-distance heuristic. All coordinates are normalized to [0, 1].

### 3. Prepare Mixed Datasets

The comparative study requires training sets where a controlled percentage of images from the *source* dataset is replaced with images from the *target* dataset. This is configured through split definition files (`0.txt`, `25.txt`, `50.txt`, `75.txt`, `100.txt`) that specify which images to swap.

Edit the `source_dataset` and `second_dataset` variables at the top of each script before running.

#### Mixed COCO datasets (for Mask R-CNN)

```bash
python prepare_datasets/create_compara_coco_datasets.py
python prepare_datasets/create_compara_coco_testsets.py
```

This creates:
- Training sets: `datasets/coco_{source}_{target}_{ratio}%/` (0%, 25%, 50%, 75%, 100%)
- Test sets: `datasets/test_coco_{source}_{target}/` with `source/`, `target/`, and `combined/` sub-splits

#### Mixed YOLO datasets

```bash
python prepare_datasets/create_compara_yolo_datasets.py
python prepare_datasets/create_yaml_for_testing.py
```

### 4. Train Models

#### YOLO11-Seg Training

Edit `train_models/train.py` to set the `data_yaml` path, model configuration, and checkpoint. Then:

```bash
python train_models/train.py
```

Training uses `imgsz=640`, `batch=2`, and `epochs=400` by default. Trained weights are saved to `runs/segment/{run_name}/weights/best.pt`.

#### Mask R-CNN Training

Edit `train_models/train_mrcnn.py` to set `source`, `target`, `ratio`, and checkpoint paths. Then:

```bash
python train_models/train_mrcnn.py
```

Training uses `IMS_PER_BATCH=2`, `BASE_LR=0.0001`, `epochs=200`, and absolute crop at 640×640. The script saves the model with the best segmentation AP to `coco_results/{DATASET_NAME}/model_best.pth`.

> **Important**: Mask R-CNN training expects a pre-trained model from the 0% mixed dataset to already exist. Train the 0% model first (using `MODEL.WEIGHTS = model_zoo.get_checkpoint_url(...)` — uncomment the relevant line in `train_mrcnn.py`), then use its `model_best.pth` for fine-tuning at other ratios.

### 5. Benchmark and Compare Models

#### Single Model Evaluation

Edit the configuration block at the top of the script (MODEL_PATH, DATASET_NAME, test directories), then:

```bash
# YOLO: reports mask mAP@50, mAP@50-95, and instance mIoU
python test_models/benchmark.py

# Mask R-CNN: reports mask mAP (AP, AP50, AP75) and instance mIoU
python test_models/benchmark_mrcnn.py
```

Both scripts compute:
- **Mask mAP**: using the framework's built-in COCO evaluator
- **Instance mIoU**: mean IoU over greedily matched prediction-GT mask pairs (bbox IoU ≥ 0.5 for YOLO, mask IoU ≥ 0.5 for Mask R-CNN)

#### Batch Comparative Testing

These scripts iterate over all source/target/ratio combinations (4 datasets × 4 targets × 5 ratios = 80 combinations) and evaluate each trained model on source-only, target-only, and combined test splits:

```bash
# YOLO: outputs yolo_result_{source}.json
python test_models/comparative_test.py

# Mask R-CNN: outputs coco_result_{source}.json
python test_models/comparative_test_mrcnn.py
```

Results are saved as JSON files mapping `source → ratio → target → [metric_string]`, where the metric string contains `mIoU | mAP50 | mAP` for YOLO and `mIoU | mAP50 | mAP` for Mask R-CNN.

---

## License

This project is licensed under the terms included in the EVERYFISH project.

---

## Acknowledgement

<img src="EN_FundedbytheEU_RGB_POS.png" alt="Funded by the European Union" width="400"/>

This project has received funding from the European Union's Horizon Europe research and innovation programme under grant agreement No 101059892 (EVERYFISH).
---

## Contact

For questions about this codebase, please contact Hang Zhou (hang.zhou@uea.ac.uk).
