# trained using the local ultralytics
from ultralytics import YOLO
from pathlib import Path

data_yaml = 'agnostic_autofish_uea_100.yaml'
run_name = Path(data_yaml).stem
# Load a model
#model = YOLO("yolo12s-seg.yaml")
#model = YOLO("yolo12s-seg.yaml").load('yolo12s.pt')  # load a pretrained model (recommended for training)
model = YOLO("yolo12s-seg.yaml").load('runs/segment/agnostic_autofish_uea_0/weights/best.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data=data_yaml, name=run_name, batch=2, epochs=400, imgsz=640)
