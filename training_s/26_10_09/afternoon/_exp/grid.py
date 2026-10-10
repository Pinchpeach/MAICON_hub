"""사전학습 yolov8n(COCO) 추론만으로 수량 예측: imgsz x conf 격자를 공개 학습 30장으로 평가(MAE, 완전일치).
원시 예측(conf>=0.03)을 캐시해 두고 임계값은 후처리로 바꾼다."""
import sys, os, pickle, numpy as np, pandas as pd
from ultralytics import YOLO
p = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else 0
tl = pd.read_csv(f"{p}/data/train_labels.csv"); m = YOLO(f"{p}/yolov8n.pt")
GROUPS = {"person": [0], "vehicle": [2, 5, 7]}; targets = [c[:-6] for c in tl.columns if c.endswith("_count")]
paths = [f"{p}/data/train_images/{f}" for f in tl.filename]; cache = {}
for sz in (416, 640, 960, 1280):
    for iou in (0.7, 0.5):
        res = m.predict(paths, imgsz=sz, conf=0.03, iou=iou, device=dev, verbose=False, max_det=300)
        cache[(sz, iou)] = [(r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(), r.boxes.xyxyn.cpu().numpy()) for r in res]
pickle.dump(cache, open(f"_exp/{p}_raw.pkl", "wb"))
T = np.round(np.arange(0.10, 0.61, 0.05), 2)
for t in targets:
    y = tl[f"{t}_count"].values; print(f"=== {p} {t}: 정답 평균 {y.mean():.2f}")
    for (sz, iou), R in cache.items():
        row = []
        for th in T:
            pr = np.array([int((np.isin(c, GROUPS[t]) & (s >= th)).sum()) for c, s, _ in R]); row.append((np.abs(pr - y).mean(), (pr == y).mean(), (pr - y).mean()))
        b = int(np.argmin([r[0] for r in row]))
        print(f"imgsz{sz} iou{iou} 최소 MAE {row[b][0]:.3f}@{T[b]} (일치 {row[b][1]:.2f}, 편향 {row[b][2]:+.2f}) | " + " ".join(f"{th}:{r[0]:.2f}" for th, r in zip(T, row)))
