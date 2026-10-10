"""전이학습 모델 평가. 정답이 없어서 (a) 검증 11장의 드론 검출, (b) 테스트 300장에서 드론 검출이 기준선 종류와 어떻게 겹치는지를 본다.
사용: python _eval_tl.py <실험 이름> [imgsz]"""
import sys, os, glob, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from ultralytics import YOLO
name = sys.argv[1]; imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 640; dev = 0 if torch.cuda.is_available() else "cpu"
w = f"runs_tl/{name}/weights/best.pt"; w = w if os.path.exists(w) else f"runs_tl/{name}/weights/last.pt"; m = YOLO(w)
AO = "data/aerial_objects"; tf = sorted(os.listdir(AO)); fn = f"_tl_pred_{name}_{imgsz}.pkl"
if os.path.exists(fn): P = pickle.load(open(fn, "rb"))
else:
    P = {}
    for i in range(0, len(tf), 10):
        for f, r in zip(tf[i:i + 10], m.predict([os.path.join(AO, x) for x in tf[i:i + 10]], imgsz=imgsz, conf=0.05, iou=0.6, device=dev, verbose=False)):
            P[f] = (r.boxes.conf.cpu().numpy(), r.boxes.xyxyn.cpu().numpy())
    pickle.dump(P, open(fn, "wb"))
B = pd.read_csv("_p2_baseline_detail.csv").set_index("filename")
print(f"== {name} (imgsz {imgsz}) 테스트 300장 ==")
for thr in (0.25, 0.5, 0.7):
    has = pd.Series({f: bool((P[f][0] >= thr).any()) for f in tf}); n = pd.Series({f: int((P[f][0] >= thr).sum()) for f in tf})
    print("  conf %.2f: 드론 검출 이미지 %3d장 | 기준선 종류별: drone %d/%d, airplane %d/%d, bird %d/%d | 드론 수 분포(검출된 이미지): %s" % (
        thr, has.sum(), has[B.kind == "drone"].sum(), (B.kind == "drone").sum(), has[B.kind == "airplane"].sum(), (B.kind == "airplane").sum(), has[B.kind == "bird"].sum(), (B.kind == "bird").sum(), n[has].value_counts().sort_index().to_dict()))
