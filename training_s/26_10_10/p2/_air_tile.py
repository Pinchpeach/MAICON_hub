"""비행기로 판정된 이미지(v3 판정, 답지 미사용)에 대해 기본 yolov8s 를 타일(2x2, 3x3)·960·TTA 로 다시 추론해 후보 박스를 저장한다. 모델은 그대로다."""
import os, pickle, numpy as np, pandas as pd, torch
from PIL import Image
from ultralytics import YOLO
AO = "data/aerial_objects"; cols = ["airplane", "bird", "drone"]; dev = 0 if torch.cuda.is_available() else "cpu"
S = pd.read_csv("submission.csv").set_index("filename"); files = [f for f in S.index if S.loc[f, "airplane"] > 0]; print("비행기 판정 이미지", len(files))
m = YOLO("yolov8s.pt"); OUT = {}
def tiles(W, H, g, ov=0.25):
    tw, th = W / (g - (g - 1) * ov), H / (g - (g - 1) * ov); out = []
    for iy in range(g):
        for ix in range(g):
            x0, y0 = ix * tw * (1 - ov), iy * th * (1 - ov); out.append((int(round(x0)), int(round(y0)), int(round(min(x0 + tw, W))), int(round(min(y0 + th, H)))))
    return out
def run(ims, **kw):
    res = []
    for i in range(0, len(ims), 9):
        for r in m.predict(ims[i:i + 9], conf=0.05, iou=0.6, device=dev, verbose=False, **kw): res.append((r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()))
    return res
for n, f in enumerate(files):
    im = Image.open(os.path.join(AO, f)).convert("RGB"); W, H = im.size; d = {}
    for g in (2, 3):
        T = tiles(W, H, g); R = run([im.crop(t) for t in T], imgsz=640); C, Sc, B, E = [], [], [], []
        for (x0, y0, x1, y1), (c, s, b) in zip(T, R):
            for ci, si, bi in zip(c, s, b):
                tw, th = x1 - x0, y1 - y0; e = 0.01
                cut = (bi[0] <= e * tw and x0 > 0) or (bi[1] <= e * th and y0 > 0) or (bi[2] >= (1 - e) * tw and x1 < W) or (bi[3] >= (1 - e) * th and y1 < H)     # 타일 안쪽 경계에서 잘린 박스
                C.append(ci); Sc.append(si); B.append([(bi[0] + x0) / W, (bi[1] + y0) / H, (bi[2] + x0) / W, (bi[3] + y0) / H]); E.append(cut)
        d["tile%d" % g] = (np.array(C, int), np.array(Sc), np.array(B).reshape(-1, 4), np.array(E, bool))
    for nm, kw in (("s960", dict(imgsz=960)), ("tta640", dict(imgsz=640, augment=True)), ("tta960", dict(imgsz=960, augment=True))):
        c, s, b = run([im], **kw)[0]; d[nm] = (c, s, b / np.array([W, H, W, H]), np.zeros(len(c), bool))
    OUT[f] = d
    if n % 20 == 0: print(n, f, {k: len(v[0]) for k, v in d.items()}, flush=True)
pickle.dump(OUT, open("_air_tile.pkl", "wb")); print("saved")
