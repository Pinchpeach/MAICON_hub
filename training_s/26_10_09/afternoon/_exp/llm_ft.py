"""축소·확대 변형으로 늘린 학습 세트(의사 박스 포함)로 yolov8n 을 20에폭 미세조정, 5겹 OOF 로 평가(2번).
변형: 전체 축소 0.75 / 0.5(회색 캔버스), 부분 확대(1.4배 자르기, 잘리는 객체가 있는 자르기는 폐기). 박스는 함께 변환한다.
사용: python _exp/aug_ft.py <epochs> <imgsz> <freeze> <lr0> <tag>"""
import sys, os, shutil, pickle, numpy as np, pandas as pd, yaml
sys.path.insert(0, "_exp")
from lib import _collect, _iou_inter
from merge2 import merge2
from ultralytics import YOLO
from sklearn.model_selection import KFold
from PIL import Image

epochs, imgsz, freeze, lr0, tag = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
p = "problem02"; ROOT = os.path.abspath(f"_exp/ft/{p}_llm_{tag}"); tl = pd.read_csv(f"{p}/data/train_labels.csv"); V = pickle.load(open(f"_exp/{p}_views.pkl", "rb"))
ALL7 = ["full640", "flip640", "full960", "flip960", "full1280", "flip1280", "tile960"]; KEYS = ["full960", "full1280", "tile960"]; rng = np.random.default_rng(0)

def pseudo_boxes(f):
    """정답 수량만큼 융합 점수 상위 군집을 박스 라벨로(정규화 xyxy)."""
    B, S, Vw = [], [], []
    for vi, k in enumerate(ALL7):
        b, s, c = V[("train_images", f)][k]; mk = c == 0; B.append(b[mk]); S.append(s[mk]); Vw.append(np.full(int(mk.sum()), vi))
    B, S, Vw = map(np.concatenate, (B, S, Vw)); reps, mem = [], []
    for i in np.argsort(-S):
        if reps:
            io, ct = _iou_inter(B[i], B[reps]); j = np.where((io >= 0.5) | (ct >= 0.8))[0]
            if len(j): mem[j[0]].append(i); continue
        reps.append(i); mem.append([i])
    sc = []
    for mm in mem:
        best = {}
        for i in mm: best[Vw[i]] = max(best.get(Vw[i], 0), S[i])
        sc.append(sum(best.values()) / len(ALL7))
    order = np.argsort(-np.array(sc))[: int(tl.loc[tl.filename == f, "person_count"].iloc[0])]
    return [B[reps[i]] for i in order]

def write(d, sub, name, im, boxes):
    im.save(f"{d}/images/{sub}/{name}.jpg", quality=92)
    open(f"{d}/labels/{sub}/{name}.txt", "w").write("".join(f"0 {(b[0]+b[2])/2:.6f} {(b[1]+b[3])/2:.6f} {b[2]-b[0]:.6f} {b[3]-b[1]:.6f}\n" for b in boxes))

def variants(im, boxes):
    """(이름, 이미지, 박스) 목록: 축소 2종 + 부분 확대 최대 2개."""
    W, H = im.size; out = []
    for s in (0.75, 0.5):
        w2, h2 = int(W * s), int(H * s); ox, oy = int(rng.integers(0, W - w2 + 1)), int(rng.integers(0, H - h2 + 1)); cv = Image.new("RGB", (W, H), (114, 114, 114)); cv.paste(im.resize((w2, h2), Image.BILINEAR), (ox, oy))
        out.append((f"sh{int(s*100)}", cv, [np.array([(b[0] * w2 + ox) / W, (b[1] * h2 + oy) / H, (b[2] * w2 + ox) / W, (b[3] * h2 + oy) / H]) for b in boxes]))
    got = 0
    for _ in range(40):
        if got >= 2: break
        z = rng.uniform(1.3, 1.6); cw, ch = 1 / z, 1 / z; x0, y0 = rng.uniform(0, 1 - cw), rng.uniform(0, 1 - ch); keep, ok = [], True
        for b in boxes:
            ix1, iy1, ix2, iy2 = max(b[0], x0), max(b[1], y0), min(b[2], x0 + cw), min(b[3], y0 + ch); vis = max(0, ix2 - ix1) * max(0, iy2 - iy1) / ((b[2] - b[0]) * (b[3] - b[1]) + 1e-12)
            if vis >= 0.9: keep.append(np.array([(ix1 - x0) / cw, (iy1 - y0) / ch, (ix2 - x0) / cw, (iy2 - y0) / ch]))
            elif vis > 0.05: ok = False; break
        if ok and keep:
            cr = im.crop((int(x0 * W), int(y0 * H), int((x0 + cw) * W), int((y0 + ch) * H))).resize((W, H), Image.BILINEAR); out.append((f"zi{got}", cr, keep)); got += 1
    return out

def views_of(model, arr):
    Hh, Ww = arr.shape[:2]; kw = dict(conf=0.03, iou=0.7, device=0, verbose=False, max_det=300); out = {}
    for sz in (960, 1280): out[f"full{sz}"] = _collect(model.predict(arr, imgsz=sz, **kw)[0])
    tw, th = int(Ww * 0.625), int(Hh * 0.625); parts = []
    for x in (0, Ww - tw):
        for yy in (0, Hh - th): parts.append(_collect(model.predict(arr[yy:yy + th, x:x + tw].copy(), imgsz=960, **kw)[0], x / Ww, yy / Hh, tw / Ww, th / Hh))
    out["tile960"] = tuple(np.concatenate([q[i] for q in parts]) for i in range(3)); return out
def shrink_arr(arr, s=0.5):
    Hh, Ww = arr.shape[:2]; cv = Image.new("RGB", (Ww, Hh), (114, 114, 114)); cv.paste(Image.fromarray(arr[:, :, ::-1]).resize((int(Ww * s), int(Hh * s)), Image.BILINEAR), (int(Ww * (1 - s) / 2), int(Hh * (1 - s) / 2))); return np.asarray(cv)[:, :, ::-1].copy()

names = YOLO(f"{p}/yolov8n.pt").names; shutil.rmtree(ROOT, ignore_errors=True); files = tl.filename.values; y = tl.person_count.values
import json
LL = json.load(open("_exp/llm/labels.json")); PB = {f: [np.array(b) for b in LL[f]["boxes"]] for f in files}; VALID = {f for f in files if LL[f]["valid"]}; res = {"orig": {}, "sh50": {}}; base = {"orig": {}, "sh50": {}}; pre = YOLO(f"{p}/yolov8n.pt"); n_train = []
for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(files)):
    d = f"{ROOT}/fold{k}"
    for sub in ("train", "val"): os.makedirs(f"{d}/images/{sub}"); os.makedirs(f"{d}/labels/{sub}")
    n = 0
    for f in files[a]:
        if f not in VALID: continue
        im = Image.open(f"{p}/data/train_images/{f}").convert("RGB"); write(d, "train", f[:-4], im, PB[f]); n += 1
        for nm, vi, vb in ([] if os.environ.get("NOAUG") else variants(im, PB[f])): write(d, "train", f"{f[:-4]}_{nm}", vi, vb); n += 1
    for f in files[b]: write(d, "val", f[:-4], Image.open(f"{p}/data/train_images/{f}").convert("RGB"), PB[f])
    n_train.append(n); yaml.safe_dump({"path": d, "train": "images/train", "val": "images/val", "names": {int(i): nn for i, nn in names.items()}}, open(f"{d}/data.yaml", "w"))
    YOLO(f"{p}/yolov8n.pt").train(data=f"{d}/data.yaml", epochs=epochs, imgsz=imgsz, batch=8, freeze=freeze, lr0=lr0, lrf=0.1, optimizer="AdamW", warmup_epochs=1, patience=epochs, workers=0, device=0,
                                 project=d, name="run", exist_ok=True, plots=False, verbose=False, seed=0, val=False, close_mosaic=5)
    ft = YOLO(f"{d}/run/weights/last.pt")
    for f in files[b]:
        arr = np.asarray(Image.open(f"{p}/data/train_images/{f}").convert("RGB"))[:, :, ::-1].copy()
        for nm, ar in (("orig", arr), ("sh50", shrink_arr(arr))):
            res[nm][f] = merge2(views_of(ft, ar), KEYS, "person", 0.5, 0.7, 2, 0.0)
            if k == 0 or f not in base[nm]: base[nm][f] = merge2(views_of(pre, ar), KEYS, "person", 0.5, 0.7, 2, 0.0)
    print("fold", k, "train images", n, flush=True)
pickle.dump((res, base, list(files), list(y)), open(f"{ROOT}/oof.pkl", "wb"))
TH = np.round(np.arange(0.05, 0.61, 0.025), 3)
print(f"RESULT 설정 {tag}: epochs {epochs}, imgsz {imgsz}, freeze {freeze}, lr0 {lr0} | 폴드당 학습 이미지 {n_train} (원본 24장)")
for nm, lab in (("orig", "원본 30장"), ("sh50", "0.5 축소본 30장(작은 사람)")):
    for name, S in (("사전학습(기준)", base[nm]), ("미세조정 OOF  ", res[nm])):
        mae = [np.mean([abs(max(int((S[f] >= t).sum()), 4) - yy) for f, yy in zip(files, y)]) for t in TH]; kk = int(np.argmin(mae))
        print(f"RESULT {lab} {name}: 최저 MAE {mae[kk]:.3f}@{TH[kk]} | 임계 0.15에서 {mae[list(TH).index(0.15)]:.3f} | " + " ".join(f"{t}:{m_:.2f}" for t, m_ in list(zip(TH, mae))[2:14:2]))
