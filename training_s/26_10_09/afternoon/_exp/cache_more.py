import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "_exp")
from lib import _collect
from ultralytics import YOLO
from PIL import Image
p = sys.argv[1]; m = YOLO(f"{p}/yolov8n.pt"); V = pickle.load(open(f"_exp/{p}_views.pkl", "rb"))
kw = dict(conf=0.03, iou=0.7, device=0, verbose=False, max_det=300)
for (split, f), views in V.items():
    arr = np.asarray(Image.open(f"{p}/data/{split}/{f}").convert("RGB"))[:, :, ::-1]; H, W = arr.shape[:2]
    views["full1600"] = _collect(m.predict(arr, imgsz=1600, **kw)[0])
    views["flip1600"] = _collect(m.predict(arr[:, ::-1].copy(), imgsz=1600, **kw)[0], flip=True)
    for name, n, frac, sz in (("tile3x3", 3, 0.45, 960), ("tile2x2f", 2, 0.625, 960)):
        tw, th = int(W * frac), int(H * frac); parts = []
        xs = np.linspace(0, W - tw, n).astype(int); ys = np.linspace(0, H - th, n).astype(int)
        for x in xs:
            for y in ys:
                crop = arr[y:y + th, x:x + tw].copy()
                if name == "tile2x2f":   # 2x2 타일의 좌우 반전
                    b, s, c = _collect(m.predict(crop[:, ::-1].copy(), imgsz=sz, **kw)[0], flip=True); b[:, [0, 2]] = x / W + b[:, [0, 2]] * tw / W; b[:, [1, 3]] = y / H + b[:, [1, 3]] * th / H; parts.append((b, s, c))
                else: parts.append(_collect(m.predict(crop, imgsz=sz, **kw)[0], x / W, y / H, tw / W, th / H))
        views[name] = tuple(np.concatenate([q[i] for q in parts]) for i in range(3))
pickle.dump(V, open(f"_exp/{p}_views.pkl", "wb")); print(p, "views:", list(next(iter(V.values())).keys()))
