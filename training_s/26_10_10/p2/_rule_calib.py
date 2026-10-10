"""정답이 있는 드론 58장에서 기본 yolov8s(COCO)의 비행기·새 점수 분포를 재서, 종류 판정 임계값의 근거로 쓴다."""
import os, pickle, numpy as np, pandas as pd, torch
from ultralytics import YOLO
TL = "data/transfer_learning"; fs = sorted(f for f in os.listdir(TL) if f.lower().endswith((".jpg", ".jpeg", ".png"))); dev = 0 if torch.cuda.is_available() else "cpu"
fn = "_base_on_transfer.pkl"; m = YOLO("yolov8s.pt"); AIR, BIRD = 4, 14
if os.path.exists(fn): R = pickle.load(open(fn, "rb"))
else:
    R = {}
    for i in range(0, len(fs), 10):
        for f, r in zip(fs[i:i + 10], m.predict([os.path.join(TL, x) for x in fs[i:i + 10]], imgsz=640, conf=0.05, iou=0.6, device=dev, verbose=False)):
            R[f] = (r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy())
    pickle.dump(R, open(fn, "wb"))
rows = []
for f in fs:
    c, s = R[f]; sa = float(s[c == AIR].max()) if (c == AIR).any() else 0.0; sb = float(s[c == BIRD].max()) if (c == BIRD).any() else 0.0
    top = int(c[np.argmax(s)]) if len(s) else -1; rows.append(dict(f=f, sa=sa, sb=sb, sp=max(sa, sb), top=m.names.get(top, "없음"), n_air=int(((c == AIR) & (s >= .25)).sum())))
T = pd.DataFrame(rows)
print("== 정답이 있는 진짜 드론 %d장에 대한 기본 yolov8s 의 비행기·새 최고 점수(sp) ==" % len(T))
for t in (0.1, 0.25, 0.35, 0.5, 0.6, 0.7, 0.8): print("  sp >= %.2f 인 드론 이미지: %2d / %d (%.0f%%)" % (t, (T.sp >= t).sum(), len(T), 100 * (T.sp >= t).mean()))
print("  sp 분위 (50%%, 75%%, 90%%): %s" % np.percentile(T.sp, [50, 75, 90]).round(2).tolist())
print("  가장 높은 점수로 본 클래스 상위:", dict(T.top.value_counts().head(6)))
print("  비행기로 본(sa >= .5) 드론 이미지 %d장, 새로 본(sb >= .5) %d장" % ((T.sa >= .5).sum(), (T.sb >= .5).sum()))
print("  sp 높은 드론 이미지:", T.sort_values("sp", ascending=False).head(8)[["f", "sa", "sb"]].round(2).values.tolist())
S = pd.read_csv("_rule_scores.csv")
print("\n== 같은 임계값으로 본 테스트 300장 (기본 모델 sp) ==")
for t in (0.25, 0.35, 0.5, 0.6, 0.7): print("  sp >= %.2f: 테스트 %3d장 (%.0f%%) | 드론 58장 중 %2d장 (%.0f%%)" % (t, (S.sp >= t).sum(), 100 * (S.sp >= t).mean(), (T.sp >= t).sum(), 100 * (T.sp >= t).mean()))
