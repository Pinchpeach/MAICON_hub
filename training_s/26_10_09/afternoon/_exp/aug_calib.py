"""변형으로 늘린 학습 세트(수량이 보존되는 변형만): 전체 축소(0.75, 0.5, 회색 캔버스), 전체 확대(1.5배)로 수량 라벨을 그대로 쓰는 (이미지, 수량) 쌍을 늘린다.
(1) 변형별 MAE(작은 사람에 대한 강건성) (2) 늘린 세트로 임계값·규칙을 고르면 원본 평가가 좋아지는지(원본 이미지 단위 LOO)."""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "_exp")
from lib import _collect
from merge2 import merge2
from ultralytics import YOLO
from PIL import Image
p = "problem02"; tl = pd.read_csv(f"{p}/data/train_labels.csv"); y = tl.person_count.values; m = YOLO(f"{p}/yolov8n.pt"); kw = dict(conf=0.03, iou=0.7, device=0, verbose=False, max_det=300)
KEYS = ["full960", "full1280", "tile960"]; rng = np.random.default_rng(0)
def views_of(arr):
    H, W = arr.shape[:2]; out = {}
    for sz in (960, 1280): out[f"full{sz}"] = _collect(m.predict(arr, imgsz=sz, **kw)[0])
    tw, th = int(W * 0.625), int(H * 0.625); parts = []
    for x in (0, W - tw):
        for yy in (0, H - th): parts.append(_collect(m.predict(arr[yy:yy + th, x:x + tw].copy(), imgsz=960, **kw)[0], x / W, yy / H, tw / W, th / H))
    out["tile960"] = tuple(np.concatenate([q[i] for q in parts]) for i in range(3)); return out
def shrink(arr, s):
    H, W = arr.shape[:2]; im = Image.fromarray(arr[:, :, ::-1]).resize((int(W * s), int(H * s)), Image.BILINEAR); canvas = Image.new("RGB", (W, H), (114, 114, 114))
    canvas.paste(im, (int(rng.integers(0, W - im.size[0] + 1)), int(rng.integers(0, H - im.size[1] + 1)))); return np.asarray(canvas)[:, :, ::-1].copy()
def enlarge(arr, s):
    H, W = arr.shape[:2]; return np.asarray(Image.fromarray(arr[:, :, ::-1]).resize((int(W * s), int(H * s)), Image.BICUBIC))[:, :, ::-1].copy()
VAR = {"orig": lambda a: a, "shrink0.75": lambda a: shrink(a, 0.75), "shrink0.5": lambda a: shrink(a, 0.5), "enlarge1.5": lambda a: enlarge(a, 1.5)}
S = {k: [] for k in VAR}
for f in tl.filename:
    a = np.asarray(Image.open(f"{p}/data/train_images/{f}").convert("RGB"))[:, :, ::-1].copy()
    for k, fn in VAR.items(): S[k].append(merge2(views_of(fn(a)), KEYS, "person", 0.5, 0.7, 2, 0.0))
pickle.dump(S, open("_exp/p02_aug_scores.pkl", "wb"))
TH = np.round(np.arange(0.05, 0.41, 0.025), 3); cnt = lambda s, t: max(int((s >= t).sum()), 4)
E = {k: np.array([[abs(cnt(s, t) - yy) for s, yy in zip(S[k], y)] for t in TH]) for k in VAR}          # [thr, img]
print("변형별 MAE (하한 4 적용), 임계값별:")
for k in VAR: print(f"   {k:11s} 현재 임계 0.15: {E[k][list(TH).index(0.15)].mean():.3f} | 최저 {E[k].mean(1).min():.3f}@{TH[int(E[k].mean(1).argmin())]} | 편향@0.15 {np.mean([cnt(s, 0.15) - yy for s, yy in zip(S[k], y)]):+.2f} | " + " ".join(f"{t}:{e:.2f}" for t, e in list(zip(TH, E[k].mean(1)))[1:9]))
# 늘린 세트로 임계값 선택 → 원본에서 평가 (원본 이미지 단위 LOO: 그 이미지의 변형도 함께 제외)
for name, pool in (("원본만", ["orig"]), ("원본+축소0.75", ["orig", "shrink0.75"]), ("원본+축소0.75+0.5", ["orig", "shrink0.75", "shrink0.5"]), ("원본+축소 2종+확대", list(VAR))):
    Ep = np.mean([E[k] for k in pool], axis=0); k_all = int(Ep.mean(1).argmin()); loo = [E["orig"][int(np.delete(Ep, i, 1).mean(1).argmin()), i] for i in range(30)]
    print(f"   임계값을 [{name}]로 선택: 선택된 임계 {TH[k_all]} | 원본 MAE {E['orig'][k_all].mean():.3f} | 원본 LOO {np.mean(loo):.3f} | 축소0.5 에서의 MAE {E['shrink0.5'][k_all].mean():.3f}")
