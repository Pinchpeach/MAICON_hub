import sys, glob, numpy as np
from ultralytics import YOLO
w = sys.argv[1]; model = YOLO(w)
imgs = sorted(glob.glob("data/val/images/*"))
def gt(img):
    c = [0]*4
    for l in open(img.replace("images","labels").rsplit(".",1)[0]+".txt").read().splitlines():
        if l.strip(): c[int(l.split()[0])] += 1
    return tuple(c)
G = [gt(i) for i in imgs]
def acc(bx, thr): return np.mean([tuple(int(((k==c)&(s>=thr)).sum()) for c in range(4)) == g for (k,s),g in zip(bx,G)])
THR = (0.2,0.25,0.3,0.35,0.4,0.5)
for sz in (640, 800, 960):
    for iou in (0.4, 0.5, 0.6, 0.7):
        for aug in (False, True):
            r = model.predict(imgs, imgsz=sz, conf=0.05, iou=iou, device=0, verbose=False, augment=aug)
            bx = [(x.boxes.cls.cpu().numpy().astype(int), x.boxes.conf.cpu().numpy()) for x in r]
            a = {t: acc(bx, t) for t in THR}; bt = max(a, key=a.get)
            print(f"imgsz {sz} iou {iou} TTA {int(aug)} | best conf {bt} acc {a[bt]:.4f} | conf0.3 {a[0.3]:.4f}", flush=True)
