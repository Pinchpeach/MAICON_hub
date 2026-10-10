"""밝기/대비 정규화 실험: (1) 오류와 밝기·대비의 상관 (2) 전처리별 관점을 만들어 같은 융합 규칙으로 평가(임계값만 LOO)."""
import sys, pickle, numpy as np, pandas as pd, cv2
sys.path.insert(0, "_exp")
from lib import _collect, merge
from ultralytics import YOLO
from PIL import Image
p = sys.argv[1]; m = YOLO(f"{p}/yolov8n.pt"); tl = pd.read_csv(f"{p}/data/train_labels.csv"); kw = dict(conf=0.03, iou=0.7, device=0, verbose=False, max_det=300)
def autocontrast(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 99); return np.clip((a.astype(np.float32) - lo) * 255.0 / max(1, hi - lo), 0, 255).astype(np.uint8)
def gamma(a):
    g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).mean() / 255.0; gm = np.log(0.45) / np.log(max(g, 1e-3)); gm = float(np.clip(gm, 0.4, 2.5))
    return (255 * (a / 255.0) ** gm).astype(np.uint8)
def clahe(a):
    lab = cv2.cvtColor(a, cv2.COLOR_BGR2LAB); lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0]); return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
def equalize(a):
    ycc = cv2.cvtColor(a, cv2.COLOR_BGR2YCrCb); ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0]); return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
def sharpen(a):
    bl = cv2.GaussianBlur(a, (0, 0), 2.0); return cv2.addWeighted(a, 1.6, bl, -0.6, 0)
PRE = {"orig": lambda a: a, "autocontrast": autocontrast, "gamma": gamma, "clahe": clahe, "equalize": equalize, "sharpen": sharpen}
def views_of(arr):
    H, W = arr.shape[:2]; out = {}
    for sz in (640, 960, 1280):
        out[f"full{sz}"] = _collect(m.predict(arr, imgsz=sz, **kw)[0]); out[f"flip{sz}"] = _collect(m.predict(arr[:, ::-1].copy(), imgsz=sz, **kw)[0], flip=True)
    tw, th = int(W * 0.625), int(H * 0.625); parts = []
    for x in (0, W - tw):
        for y in (0, H - th): parts.append(_collect(m.predict(arr[y:y + th, x:x + tw].copy(), imgsz=960, **kw)[0], x / W, y / H, tw / W, th / H))
    out["tile960"] = tuple(np.concatenate([q[i] for q in parts]) for i in range(3)); return out
V = {k: {} for k in PRE}; stats = []
for f in tl.filename:
    a = np.asarray(Image.open(f"{p}/data/train_images/{f}").convert("RGB"))[:, :, ::-1].copy(); g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    stats.append((g.mean(), g.std(), (g < 40).mean(), cv2.Laplacian(g, cv2.CV_64F).var()))
    for k, fn in PRE.items(): V[k][f] = views_of(fn(a))
pickle.dump((V, stats), open(f"_exp/{p}_bright.pkl", "wb"))
