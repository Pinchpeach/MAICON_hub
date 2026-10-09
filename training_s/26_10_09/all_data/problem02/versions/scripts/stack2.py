import sys, os, warnings, time
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from cv2 import make_features_v2, tr, CL
F = make_features_v2(tr).values; y = tr.flight_state.values; uid = tr.uid.values
W = np.array([1, 3, 5, 1, 1.0])
CACHE = os.environ.get("STACK_OUT", "stack_cache")

def ctx(P, u, ns=(5, 20)):
    df = pd.DataFrame(P); out = [P]
    for n in ns:
        out.append(df.groupby(u, sort=False).rolling(n, min_periods=1).mean().reset_index(level=0, drop=True).sort_index().values)
    return np.hstack(out)

def logit(P): P = np.clip(P, 1e-4, 1 - 1e-4); return np.log(P / (1 - P))

def meta_fit_predict(kind, Pa, ya, ua, Fa, Pb, ub, Fb, bal):
    sw = compute_sample_weight("balanced", ya) if bal else None
    if kind == "LR":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500)); m.fit(logit(Pa), ya, logisticregression__sample_weight=sw); return m.classes_, m.predict_proba(logit(Pb))
    if kind == "HGB_p": Xa, Xb = Pa, Pb
    elif kind == "HGB_ctx": Xa, Xb = ctx(Pa, ua), ctx(Pb, ub)
    elif kind == "HGB_ctx_feat": Xa, Xb = np.hstack([ctx(Pa, ua), Fa]), np.hstack([ctx(Pb, ub), Fb])
    m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_leaf_nodes=15, random_state=0).fit(Xa, ya, sample_weight=sw)
    return m.classes_, m.predict_proba(Xb)

res = {}
for kind in ("LR", "HGB_p", "HGB_ctx", "HGB_ctx_feat"):
    for bal in (False, True):
        P = np.zeros((len(tr), 5)); t = time.time()
        for o in pd.unique(uid):
            d = np.load(f"{CACHE}/outer_{o}.npz"); tm = d["tm"]; vm = ~tm
            cls, pr = meta_fit_predict(kind, d["inner"], y[tm], uid[tm], F[tm], d["val"], uid[vm], F[vm], bal)
            P[np.ix_(vm, [CL.index(c) for c in cls])] = pr
        res[(kind, bal)] = P
        for name, w in (("가중없음", np.ones(5)), ("ASCx3 DESCx5", W)):
            pred = np.array(CL)[(P * w).argmax(1)]; f = f1_score(y, pred, labels=CL, average=None, zero_division=0)
            print(f"stack2 {kind:13s} bal={int(bal)} {name:12s} acc {np.mean(pred == y):.4f} macroF1 {f.mean():.4f} | " + " ".join(f"{c[:4]}:{v:.3f}" for c, v in zip(CL, f)) + f" ({time.time() - t:.0f}s)", flush=True)
np.save(f"{CACHE}/stack2_best_probs.npy", res[("HGB_ctx", False)])
# stack1 만의 단순 argmax (비교용)
cols = [CL.index(c) for c in CL]
S1 = np.zeros((len(tr), 5))
for o in pd.unique(uid):
    d = np.load(f"{CACHE}/outer_{o}.npz"); S1[~d["tm"]] = d["val"].reshape(len(d["val"]), -1, 5).mean(1)
pred = np.array(CL)[S1.argmax(1)]; f = f1_score(y, pred, labels=CL, average=None, zero_division=0)
print(f"stack1만(RF·HGB 평균 argmax)  acc {np.mean(pred == y):.4f} macroF1 {f.mean():.4f} | " + " ".join(f"{c[:4]}:{v:.3f}" for c, v in zip(CL, f)))
