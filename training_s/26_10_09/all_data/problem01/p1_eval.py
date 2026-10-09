"""검증 160장 exact-count 정확도: conf 격자 x (imgsz, agnostic, TTA). 사용: python p1_eval.py <weights...>"""
import glob, sys, numpy as np
from ultralytics import YOLO
T = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)
imgs = sorted(glob.glob("data/val/images/*"))
def gt(i):
    c = [0]*4
    for l in open(i.replace("images", "labels").rsplit(".", 1)[0] + ".txt").read().splitlines():
        if l.strip(): c[int(l.split()[0])] += 1
    return tuple(c)
G = [gt(i) for i in imgs]
cnt = lambda k, s, t: tuple(int(((k == c) & (s >= t)).sum()) for c in range(4))
def boxes(m, **kw):
    out = []
    for i in range(0, len(imgs), 32):
        out += [(x.boxes.cls.cpu().numpy().astype(int), x.boxes.conf.cpu().numpy()) for x in m.predict(imgs[i:i+32], conf=0.05, device=0, verbose=False, **kw)]
    return out
if __name__ == "__main__":
    for w in sys.argv[1:]:
        m = YOLO(w)
        for sz in (640, 800, 960):
            for ag in (False, True):
                bx = boxes(m, imgsz=sz, iou=0.6, agnostic_nms=ag)
                a = {t: np.mean([cnt(k, s, t) == g for (k, s), g in zip(bx, G)]) for t in T}
                bt = max(a, key=a.get)
                print(f"{w.split('/')[-3]:14s} sz{sz} ag{int(ag)} best {a[bt]:.4f}@{bt} | " + " ".join(f"{t}:{a[t]:.3f}" for t in T), flush=True)
