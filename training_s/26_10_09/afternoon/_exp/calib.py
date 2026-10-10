"""개수 라벨로 배우는 보정 회귀: 융합 점수에서 만든 특징(여러 임계값의 개수, 점수 합)으로 정답 수량을 회귀. LOO 로 평가."""
import sys, pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "_exp")
from lib import merge
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import LeaveOneOut
VS = {"f640+960+1280": ["full640", "full960", "full1280"], "f960+1280+tile": ["full960", "full1280", "tile960"], "all7": ["full640", "flip640", "full960", "flip960", "full1280", "flip1280", "tile960"]}
TH = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
for p, tgt, fam, thr in [("problem02", "person", ("f960+1280+tile", 0.5, 0.7, 2), 0.15), ("problem04", "person", ("all7", 0.5, 0.8, 1), 0.125), ("problem04", "vehicle", ("f640+960+1280", 0.5, 0.7, 2), 0.175)]:
    tl = pd.read_csv(f"{p}/data/train_labels.csv"); V = pickle.load(open(f"_exp/{p}_views.pkl", "rb")); y = tl[f"{tgt}_count"].values
    S = [merge(V[("train_images", f)], VS[fam[0]], tgt, *fam[1:]) for f in tl.filename]
    cnt = np.array([[(s >= t).sum() for t in TH] for s in S], float); soft = np.array([[s[s >= 0.05].sum(), np.sqrt(s[s >= 0.05]).sum()] for s in S])
    base = np.array([(s >= thr).sum() for s in S])
    feats = {"개수@thr 1개(선형)": base.reshape(-1, 1).astype(float), "개수 7임계값(ridge)": cnt, "개수 7 + 점수합 2(ridge)": np.hstack([cnt, soft]), "점수합만": soft[:, :1]}
    print(f"=== {p} {tgt}: 규칙(임계값만) MAE {np.abs(base - y).mean():.3f}, 편향 {(base - y).mean():+.2f}")
    for name, X in feats.items():
        for mk, lab in ((lambda: LinearRegression(), "OLS"), (lambda: Ridge(3.0), "ridge3"), (lambda: Ridge(30.0), "ridge30")):
            pr = np.zeros(len(y))
            for a, b in LeaveOneOut().split(X): pr[b] = mk().fit(X[a], y[a]).predict(X[b])
            pr = np.clip(np.round(pr), 0, None); print(f"   {name:24s} {lab:8s} LOO MAE {np.abs(pr - y).mean():.3f} 일치 {(pr == y).mean():.2f} 편향 {(pr - y).mean():+.2f}")
    # 단순 편향 보정: 예측에 상수 더하기/곱하기 (LOO)
    for name, f in (("+상수", lambda a, tr: np.round(base + np.round((y[tr] - base[tr]).mean()))), ("x비율", lambda a, tr: np.round(base * (y[tr].sum() / max(1, base[tr].sum()))))):
        pr = np.zeros(len(y))
        for i in range(len(y)):
            tr = np.delete(np.arange(len(y)), i); pr[i] = f(None, tr)[i]
        print(f"   편향 보정 {name}: LOO MAE {np.abs(pr - y).mean():.3f} 일치 {(pr == y).mean():.2f}")
