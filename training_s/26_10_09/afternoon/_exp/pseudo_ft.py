"""개수 라벨로 만든 의사 박스(융합 점수 상위 N개, N = 정답 수량)로 yolov8n 을 20에폭 미세조정하고 5겹으로 평가.
사용: python _exp/pseudo_ft.py <problem> <epochs> <imgsz> <freeze> <tag>"""
import sys, os, shutil, pickle, numpy as np, pandas as pd, yaml
sys.path.insert(0, "_exp")
from lib import raw_views, merge, GROUPS, _iou_inter
from ultralytics import YOLO
from sklearn.model_selection import KFold

p, epochs, imgsz, freeze, tag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
ROOT = os.path.abspath(f"_exp/ft/{p}_{tag}"); tl = pd.read_csv(f"{p}/data/train_labels.csv"); V = pickle.load(open(f"_exp/{p}_views.pkl", "rb"))
targets = [c[:-6] for c in tl.columns if c.endswith("_count")]
ALL7 = ["full640", "flip640", "full960", "flip960", "full1280", "flip1280", "tile960"]
FAM = {("problem02", "person"): (["full960", "full1280", "tile960"], 0.5, 0.7, 2), ("problem04", "person"): (ALL7, 0.5, 0.8, 1), ("problem04", "vehicle"): (["full640", "full960", "full1280"], 0.5, 0.7, 2)}

def clusters(views, group, nms=0.5, contain=0.8):
    """모든 관점을 합쳐 군집 대표 박스·점수·클래스를 반환(의사 라벨용)."""
    B, S, C, Vw = [], [], [], []
    for vi, k in enumerate(ALL7):
        b, s, c = views[k]; m = np.isin(c, GROUPS[group]); B.append(b[m]); S.append(s[m]); C.append(c[m]); Vw.append(np.full(int(m.sum()), vi))
    B, S, C, Vw = map(np.concatenate, (B, S, C, Vw)); order = np.argsort(-S); reps, mem = [], []
    for i in order:
        if reps:
            io, ct = _iou_inter(B[i], B[reps]); j = np.where((io >= nms) | (ct >= contain))[0]
            if len(j): mem[j[0]].append(i); continue
        reps.append(i); mem.append([i])
    out = []
    for r, m in zip(reps, mem):
        best = {}
        for i in m: best[Vw[i]] = max(best.get(Vw[i], 0), S[i])
        out.append((B[r], sum(best.values()) / len(ALL7), int(C[r])))
    return sorted(out, key=lambda t: -t[1])

def pseudo_label(f):
    lines = []
    for tgt in targets:
        n = int(tl.loc[tl.filename == f, f"{tgt}_count"].iloc[0])
        for b, s, c in clusters(V[("train_images", f)], tgt)[:n]:      # 정답 수량만큼 상위 군집을 박스 라벨로
            lines.append(f"{c} {(b[0]+b[2])/2:.6f} {(b[1]+b[3])/2:.6f} {b[2]-b[0]:.6f} {b[3]-b[1]:.6f}")
    return lines

names = YOLO(f"{p}/yolov8n.pt").names; shutil.rmtree(ROOT, ignore_errors=True)
files = tl.filename.values; res = {t: {} for t in targets}; TH = np.round(np.arange(0.05, 0.61, 0.025), 3)
short = sum(max(0, int(tl.loc[tl.filename == f, f"{t}_count"].iloc[0]) - len(clusters(V[("train_images", f)], t))) for f in files for t in targets)
print("의사 박스가 정답 수량보다 모자란 개수 합:", short)
for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(files)):
    d = f"{ROOT}/fold{k}"
    for sub, idx in (("train", a), ("val", b)):
        os.makedirs(f"{d}/images/{sub}"); os.makedirs(f"{d}/labels/{sub}")
        for f in files[idx]:
            shutil.copy(f"{p}/data/train_images/{f}", f"{d}/images/{sub}/{f}"); open(f"{d}/labels/{sub}/{f[:-4]}.txt", "w").write("\n".join(pseudo_label(f)) + "\n")
    yaml.safe_dump({"path": d, "train": "images/train", "val": "images/val", "names": {int(i): n for i, n in names.items()}}, open(f"{d}/data.yaml", "w"))
    m = YOLO(f"{p}/yolov8n.pt")
    m.train(data=f"{d}/data.yaml", epochs=epochs, imgsz=imgsz, batch=8, freeze=freeze, lr0=float(os.environ.get("LR0", "0.001")), lrf=0.1, optimizer="AdamW", warmup_epochs=1, patience=epochs, workers=0, device=0,
            project=d, name="run", exist_ok=True, plots=False, verbose=False, seed=0, val=False, close_mosaic=5)
    ft = YOLO(f"{d}/run/weights/last.pt")
    for f in files[b]:
        views = raw_views(ft, f"{p}/data/train_images/{f}", 0)
        for t in targets:
            keys, nms, ct, vote = FAM[(p, t)]; res[t][f] = merge(views, keys, t, nms, ct, vote)
    print("fold", k, "done", flush=True)
pickle.dump(res, open(f"{ROOT}/oof_scores.pkl", "wb"))
for t in targets:
    y = tl[f"{t}_count"].values; keys, nms, ct, vote = FAM[(p, t)]
    base = [merge(V[("train_images", f)], keys, t, nms, ct, vote) for f in files]
    for name, S in (("사전학습(기준)", base), (f"미세조정 {tag} OOF", [res[t][f] for f in files])):
        mae = [np.abs(np.array([(s >= th).sum() for s in S]) - y).mean() for th in TH]; k = int(np.argmin(mae))
        sm = min(np.mean(mae[max(0, i - 1):i + 2]) for i in range(len(TH)))
        print(f"{p} {t} {name:22s} 최저 MAE {mae[k]:.3f}@{TH[k]} | 고원 {sm:.3f} | " + " ".join(f"{th}:{m_:.2f}" for th, m_ in list(zip(TH, mae))[2:15:2]))
