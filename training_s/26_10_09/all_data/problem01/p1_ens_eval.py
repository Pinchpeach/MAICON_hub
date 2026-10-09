import sys, glob, numpy as np, pickle, os
from p1_ens import *
paths = sorted(glob.glob("data/val/images/*")); G = [gt_counts(p) for p in paths]
T = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6); T2 = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55)
def ev(name, raws, flip):
    n_sets = len(raws) * (2 if flip else 1)
    cl = [wbf([s for r in raws for s in r[i]], n_sets, agnostic=True) for i in range(len(paths))]
    a = {t: np.mean([counts_from(c, t) == g for c, g in zip(cl, G)]) for t in T}
    print(f"{name:34s} mean(0.3-0.55) {np.mean([a[t] for t in T2]):.4f} best {max(a.values()):.4f}@{max(a, key=a.get)} | " + " ".join(f"{t}:{a[t]:.3f}" for t in T), flush=True)
    return cl
W = sys.argv[1:]
raw = {w: predict_raw(w, paths, flip=False) for w in W}
rawf = {w: predict_raw(w, paths, flip=True) for w in W}
for w in W: ev(os.path.basename(os.path.dirname(os.path.dirname(w))) + " single", [raw[w]], False)
for w in W: ev(os.path.basename(os.path.dirname(os.path.dirname(w))) + " +flipTTA", [rawf[w]], True)
ev(f"ens{len(W)}", [raw[w] for w in W], False)
cl = ev(f"ens{len(W)}+flipTTA", [rawf[w] for w in W], True)
pickle.dump((paths, G), open("audit/_val_meta.pkl", "wb"))
