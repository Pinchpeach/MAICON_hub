import sys, json
from ultralytics import YOLO

if __name__ == "__main__":
    weights, name, imgsz, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    extra = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    kw = dict(batch=8, patience=30, workers=6, cache=True); kw.update(extra)
    YOLO(weights).train(data="data_local.yaml", epochs=epochs, imgsz=imgsz, device=0, project="training_runs", name=name,
                        seed=0, plots=False, verbose=False, exist_ok=True, **kw)
