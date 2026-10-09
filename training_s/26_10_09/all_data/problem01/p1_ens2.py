import glob, itertools, pickle, numpy as np
from p1_ens import *
paths = sorted(glob.glob("data/val/images/*")); G = [gt_counts(p) for p in paths]
T = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6); T2 = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55)
M = {"b0": ("b_fix25_s0", 640), "b1": ("b_fix25_s1", 640), "k0": ("c_cls15_s0", 640), "k1": ("c_cls15_s1", 640), "s0": ("c_sz800_s0", 800), "s1": ("c_sz800_s1", 800), "a100": ("a_fix640", 640)}
raw = {}; rawf = {}
for k, (n, sz) in M.items():
    w = f"runs/p1/{n}/weights/last.pt"; raw[k] = predict_raw(w, paths, imgsz=sz); rawf[k] = predict_raw(w, paths, imgsz=sz, flip=True)
pickle.dump((raw, rawf), open("audit/_val_raw.pkl", "wb"))
def ev(keys, flip):
    src = rawf if flip else raw; n_sets = len(keys) * (2 if flip else 1)
    cl = [wbf([s for k in keys for s in src[k][i]], n_sets, agnostic=True) for i in range(len(paths))]
    a = {t: np.mean([counts_from(c, t) == g for c, g in zip(cl, G)]) for t in T}
    return np.mean([a[t] for t in T2]), max(a.values()), max(a, key=a.get)
combos = {"b0+b1 (기준 2)": ["b0", "b1"], "b0+b1+k0+k1": ["b0", "b1", "k0", "k1"], "25에폭 6개": ["b0", "b1", "k0", "k1", "s0", "s1"], "25에폭 6개+a100": ["b0", "b1", "k0", "k1", "s0", "s1", "a100"], "b0+b1+a100": ["b0", "b1", "a100"]}
for name, ks in combos.items():
    for flip in (False, True):
        m, b, bt = ev(ks, flip); print(f"{name:20s} flip={int(flip)} mean(0.3-0.55) {m:.4f} best {b:.4f}@{bt}", flush=True)
