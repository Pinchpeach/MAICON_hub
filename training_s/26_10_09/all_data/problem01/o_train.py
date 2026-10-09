import sys, json
from ultralytics import YOLO

# 사용: python o_train.py <data yaml> <name> <imgsz> <epochs> '<추가 인자 json>'
data, name, imgsz, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
kw = dict(batch=16, patience=15); kw.update(json.loads(sys.argv[5]) if len(sys.argv) > 5 else {})
YOLO("yolov8n.pt").train(data=data, epochs=epochs, imgsz=imgsz, workers=0, device=0, project="training_runs", name=name,
                         seed=0, deterministic=True, plots=False, verbose=False, exist_ok=True, **kw)
