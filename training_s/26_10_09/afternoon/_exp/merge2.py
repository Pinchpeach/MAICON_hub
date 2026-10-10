"""융합 규칙 개선 시험: 포함도 병합에 크기 비율 조건 추가(작은 박스가 큰 박스 안에 있어도 크기가 많이 다르면 다른 객체로 본다)."""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "_exp")
from lib import GROUPS
def merge2(views, keys, group, nms_iou=0.5, contain=0.7, vote=1, min_ratio=0.0):
    B, S, V = [], [], []
    for vi, k in enumerate(keys):
        b, s, c = views[k]; m = np.isin(c, GROUPS[group]); B.append(b[m]); S.append(s[m]); V.append(np.full(int(m.sum()), vi))
    B = np.concatenate(B); S = np.concatenate(S); V = np.concatenate(V); A = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]); reps, members = [], []
    for i in np.argsort(-S):
        if reps:
            R = B[reps]; iw = np.maximum(0, np.minimum(B[i, 2], R[:, 2]) - np.maximum(B[i, 0], R[:, 0])); ih = np.maximum(0, np.minimum(B[i, 3], R[:, 3]) - np.maximum(B[i, 1], R[:, 1]))
            inter = iw * ih; ar = A[reps]; io = inter / (A[i] + ar - inter + 1e-12); ct = inter / (np.minimum(A[i], ar) + 1e-12); ratio = np.minimum(A[i], ar) / (np.maximum(A[i], ar) + 1e-12)
            j = np.where((io >= nms_iou) | ((ct >= contain) & (ratio >= min_ratio)))[0]
            if len(j): members[j[0]].append(i); continue
        reps.append(i); members.append([i])
    out = []
    for mem in members:
        best = {}
        for i in mem: best[V[i]] = max(best.get(V[i], 0), S[i])
        if len(best) >= vote: out.append(sum(best.values()) / len(keys))
    return np.array(out)
if __name__ == "__main__":
    TH = np.round(np.arange(0.05, 0.61, 0.025), 3)
    CF = [("problem02", "person", ["full960", "full1280", "tile960"], 2, 4), ("problem04", "person", ["full640", "flip640", "full960", "flip960", "full1280", "flip1280"], 1, 0), ("problem04", "vehicle", ["full640", "full960", "full1280"], 2, 0)]
    for p, tgt, keys, vote, lo in CF:
        tl = pd.read_csv(f"{p}/data/train_labels.csv"); y = tl[f"{tgt}_count"].values; V = pickle.load(open(f"_exp/{p}_views.pkl", "rb")); print(f"=== {p} {tgt} (하한 {lo})")
        for ct in (0.7, 0.8, 0.9):
            for r in (0.0, 0.1, 0.2, 0.3, 0.5):
                for vt in sorted({vote, 1, 2}):
                    S = [merge2(V[("train_images", f)], keys, tgt, 0.5, ct, vt, r) for f in tl.filename]
                    E = np.array([np.abs(np.maximum(np.array([(s >= t).sum() for s in S]), lo) - y) for t in TH]); sm = np.array([E[max(0, k - 1):k + 2].mean() for k in range(len(TH))]); k = int(np.argmin(sm))
                    loo = np.mean([E[int(np.argmin(np.delete(E, i, 1).mean(1))), i] for i in range(len(y))]); bias = (np.maximum(np.array([(s >= TH[k]).sum() for s in S]), lo) - y).mean()
                    if r == 0.0 or loo <= 0.0 or True:
                        print(f"   포함도 {ct} 크기비율>={r} vote {vt}: MAE {E[k].mean():.3f} LOO {loo:.3f} 고원 {sm[k]:.3f} thr {TH[k]} 편향 {bias:+.2f}")
