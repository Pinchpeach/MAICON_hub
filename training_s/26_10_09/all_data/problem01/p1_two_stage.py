"""2단계 방식 평가: (1) 1클래스 '차량' yolov8n 으로 대수·위치 결정 (2) 4클래스 yolov8n 점수로 클래스 부여.
사용: python p1_two_stage.py <1cls weights> <4cls weights>   → val 총 대수 정확도와 exact-count 정확도"""
import glob, sys, numpy as np
from ultralytics import YOLO
from p1_ens import gt_counts, iou_xyxy

T = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)

def raw(weights, paths, conf, iou=0.6, imgsz=640, agnostic=False):
    m = YOLO(weights); out = []
    for i in range(0, len(paths), 32):
        for r in m.predict(paths[i:i+32], imgsz=imgsz, conf=conf, iou=iou, device=0, verbose=False, agnostic_nms=agnostic, max_det=300):
            out.append((r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)))
    return out

def assign_class(box, b4, s4, c4, prior):
    """1클래스 박스에 4클래스 모델의 겹치는 박스 점수를 모아 클래스를 정한다. 겹치는 것이 없으면 사전확률 최빈 클래스."""
    if len(b4):
        io = iou_xyxy(box, b4); sel = io >= 0.5
        if sel.any():
            v = np.bincount(c4[sel], weights=s4[sel] * io[sel], minlength=4); return int(v.argmax())
        sel = io >= 0.3
        if sel.any():
            v = np.bincount(c4[sel], weights=s4[sel] * io[sel], minlength=4); return int(v.argmax())
    return int(prior.argmax())

def two_stage_counts(d1, d4, thr, prior):
    b1, s1, _ = d1; b4, s4, c4 = d4; c = [0] * 4
    for b, s in zip(b1, s1):
        if s >= thr: c[assign_class(b, b4, s4, c4, prior)] += 1
    return tuple(c)

if __name__ == "__main__":
    w1, w4 = sys.argv[1], sys.argv[2]
    paths = sorted(glob.glob("data/val/images/*")); G = [gt_counts(p) for p in paths]
    prior = np.array([510, 111, 94, 105], float)
    D1 = raw(w1, paths, 0.05); D4 = raw(w4, paths, 0.01, iou=0.7); D4a = raw(w4, paths, 0.05, agnostic=True)
    tot1 = {t: np.mean([int((d[1] >= t).sum()) == sum(g) for d, g in zip(D1, G)]) for t in T}
    tot4 = {t: np.mean([int((d[1] >= t).sum()) == sum(g) for d, g in zip(D4a, G)]) for t in T}
    ex2 = {t: np.mean([two_stage_counts(d1, d4, t, prior) == g for d1, d4, g in zip(D1, D4, G)]) for t in T}
    ex4 = {t: np.mean([tuple(int(((d[2] == c) & (d[1] >= t)).sum()) for c in range(4)) == g for d, g in zip(D4a, G)]) for t in T}
    f = lambda a: f"최고 {max(a.values()):.4f}@{max(a, key=a.get)} | " + " ".join(f"{t}:{a[t]:.3f}" for t in T)
    print("총 대수 정확도  1클래스 :", f(tot1)); print("총 대수 정확도  4클래스 :", f(tot4))
    print("exact-count    2단계   :", f(ex2)); print("exact-count    4클래스 :", f(ex4))
    bt = max(tot1, key=tot1.get)
    print("객체 수 합계: 정답", sum(map(sum, G)), "| 1클래스 예측", int(sum((d[1] >= bt).sum() for d in D1)), f"(@{bt}) | 4클래스 예측", int(sum((d[1] >= max(tot4, key=tot4.get)).sum() for d in D4a)))
