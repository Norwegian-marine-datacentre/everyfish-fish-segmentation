# This is repo for EVERYFISH AI module -- species-agnostic fish segmentation

UEA train and test YOLO11-Seg(Ultralytics) and Mask RCNN(detectron2) on CatchMonitor, CatchHawk, CatchWam and AutoFish

## Data preparation

- Download datasets in EVERYFISH-COCO format
- Use tools in 'convertors_everyfishcoco2coco' for Mask RCNN required data format, and 'convertors_everyfishcoco2yolo'
for YOLO training.
- To prepare mixed dataset for comparative study, use code in 'prepare_datasets'.

## Model training and validation

- 'train.py' for YOLO training, 'train_mrcnn.py' for Mask RCNN.
- To benchmark individual model either for YOLO or Mask RCNN use 'benchmark.py' and 'benchmark_mcrnn.py'
- To do batch-benchmark, use 'comparative_test.py' and 'comparative_test_mrcnn.py'. It will generate a json that summarize 
multiple metrics from different models into one json.

