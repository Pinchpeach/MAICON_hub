"""yolov8n 전이학습(이번 세션 규칙: yolov8n 만). 사용: python p1_train.py <data yaml> <name> <imgsz> <epochs> '<추가 인자 json>'"""
import sys, json, os
from ultralytics import YOLO
if __name__ == "__main__":
    data, name, imgsz, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    kw = dict(batch=16, patience=100, workers=4, cache=True, cos_lr=True, close_mosaic=15, seed=0, deterministic=False)
    kw.update(json.loads(sys.argv[5]) if len(sys.argv) > 5 else {})
    YOLO("yolov8n.pt").train(data=data, epochs=epochs, imgsz=imgsz, device=0, project=os.path.abspath("runs/p1"), name=name,
                             plots=False, verbose=False, exist_ok=True, **kw)
