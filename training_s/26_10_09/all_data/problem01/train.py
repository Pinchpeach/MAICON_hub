import sys, json
from ultralytics import YOLO
weights, name, imgsz, epochs, dev = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
extra = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
kw = dict(batch=16, patience=10); kw.update(extra)
YOLO(weights).train(data="data_local.yaml", epochs=epochs, imgsz=imgsz, workers=0, device=dev, project="training_runs", name=name,
                    seed=0, deterministic=True, plots=False, verbose=False, exist_ok=True, **kw)
