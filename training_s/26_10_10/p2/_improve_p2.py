"""v3 후보 A(비행기·새 수량을 640+1280 융합으로), B(드론 수량 포함도 병합), C(A+B) 를 v2 와 같은 종류 판정 위에서 평가한다. 평가 전용, 1회."""
import os, pickle, numpy as np, pandas as pd, torch
from PIL import Image
from ultralytics import YOLO
AIR, BIRD = 4, 14; TD, TP, TP_LO, AR_T = 0.35, 0.5, 0.1, 2.2; AO = "data/aerial_objects"; dev = 0 if torch.cuda.is_available() else "cpu"
D = pickle.load(open("_final_preds.pkl", "rb")); PD, PB = D["P_drone"], D["P_base"]; files = sorted(PB)
fn = "_base1280.pkl"
if os.path.exists(fn): P12 = pickle.load(open(fn, "rb"))
else:
    m = YOLO("yolov8s.pt"); P12 = {}
    for i in range(0, len(files), 8):
        for f, r in zip(files[i:i + 8], m.predict([os.path.join(AO, x) for x in files[i:i + 8]], imgsz=1280, conf=0.05, iou=0.6, device=dev, verbose=False)):
            P12[f] = (r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(), r.boxes.xyxyn.cpu().numpy())
    pickle.dump(P12, open(fn, "wb"))
def ic(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = iw * ih; aa = (a[2] - a[0]) * (a[3] - a[1]); bb = (b[2] - b[0]) * (b[3] - b[1]); return i / (aa + bb - i + 1e-12), i / (min(aa, bb) + 1e-12)
def mb(bx, sc, thr, it=0.5, ct=0.8):
    keep = []
    for i in np.argsort(-sc):
        if sc[i] < thr: break
        if not any(ic(bx[i], bx[j])[0] >= it or ic(bx[i], bx[j])[1] >= ct for j in keep): keep.append(i)
    return keep
def kind_of(f):
    c, s, b = PB[f]; dc, ds, db = PD[f]; W, H = Image.open(os.path.join(AO, f)).size
    sa = s[c == AIR].max() if (c == AIR).any() else 0.0; sb = s[c == BIRD].max() if (c == BIRD).any() else 0.0; sd = ds.max() if len(ds) else 0.0
    if sb >= TP and sb >= sa: return "bird"
    if sa >= TP:
        i = np.where(c == AIR)[0][np.argmax(s[c == AIR])]; ar = (b[i][2] - b[i][0]) * W / max((b[i][3] - b[i][1]) * H, 1e-6); return "airplane" if ar >= AR_T else "drone"
    if sd >= TD: return "drone"
    if sb >= TP_LO and sb >= sa: return "bird"
    if sa >= TP_LO: return "airplane"
    return "drone"
def count(f, kind, air_fuse, bird_fuse, drone_contain):
    c, s, b = PB[f]; c2, s2, b2 = P12[f]; dc, ds, db = PD[f]
    if kind in ("airplane", "bird"):
        k = AIR if kind == "airplane" else BIRD; fuse = air_fuse if kind == "airplane" else bird_fuse
        bx, sc = b[c == k], s[c == k]
        if fuse: bx, sc = np.concatenate([bx, b2[c2 == k]]) if len(bx) + (c2 == k).sum() else bx, np.concatenate([sc, s2[c2 == k]])
        return max(len(mb(bx, sc, 0.25)), 1)
    return max(len(mb(db, ds, TD, 0.5, 0.8 if drone_contain else 1.01)), 1)
A = pd.read_csv(r"C:\Users\User\Downloads\submission (1).csv"); cols = ["airplane", "bird", "drone"]; K = {f: kind_of(f) for f in files}
S = pd.read_csv("submission.csv"); same = sum(K[f] == cols[int(np.argmax(S.set_index("filename").loc[f, cols].values > 0))] for f in files); print("v2 종류 재현: %d / 300" % same)
def run(af, bf, dc):
    rows = []
    for f in files:
        n = count(f, K[f], af, bf, dc); rows.append(dict(filename=f, **{k: (n if K[f] == k else 0) for k in cols}))
    return pd.DataFrame(rows)
acc = lambda X: int((X[cols].values == A[cols].values).all(1).sum())
res = {"v2 (현재)": run(False, False, False), "A: 비행기만 640+1280": run(True, False, False), "A': 비행기·새 640+1280": run(True, True, False), "B: 드론 포함도 병합": run(False, False, True), "C: A' + B": run(True, True, True)}
base = acc(res["v2 (현재)"]); V2 = res["v2 (현재)"][cols].values; AK = A[cols].values
print("%-26s %-12s %-10s %s" % ("후보", "정확 일치", "v2 대비", "바뀐 이미지 중 좋아짐/나빠짐"))
for n, X in res.items():
    ch = (X[cols].values != V2).any(1); ok_new = (X[cols].values == AK).all(1); ok_old = (V2 == AK).all(1)
    better = int((ok_new & ~ok_old).sum()); worse = int((~ok_new & ok_old).sum())
    print("%-26s %3d/300 (%.1f%%) %+d장   바뀐 %d장: 좋아짐 %d / 나빠짐 %d" % (n, acc(X), 100 * acc(X) / 300, acc(X) - base, ch.sum(), better, worse))
pickle.dump({k: v for k, v in res.items()}, open("_v3_candidates.pkl", "wb"))
