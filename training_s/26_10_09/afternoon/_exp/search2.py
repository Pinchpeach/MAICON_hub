import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "_exp")
from lib import merge
p = sys.argv[1]; tl = pd.read_csv(f"{p}/data/train_labels.csv"); V = pickle.load(open(f"_exp/{p}_views.pkl", "rb"))
F = lambda *s: [f"full{x}" for x in s]; FF = lambda *s: [k for x in s for k in (f"full{x}", f"flip{x}")]
VS = {"f960+1280+tile(v1)": F(960, 1280) + ["tile960"], "f1280+1600+tile": F(1280, 1600) + ["tile960"], "f960+1280+1600+tile": F(960, 1280, 1600) + ["tile960"],
      "f960+1280+tile+tile3": F(960, 1280) + ["tile960", "tile3x3"], "f1280+tile+tile3": F(1280) + ["tile960", "tile3x3"], "f960+1280+tile+tileF": F(960, 1280) + ["tile960", "tile2x2f"],
      "ff960+1280+tile+tileF": FF(960, 1280) + ["tile960", "tile2x2f"], "f640+960+1280(v1)": F(640, 960, 1280), "f640..1600": F(640, 960, 1280, 1600), "ff640+960+1280": FF(640, 960, 1280),
      "all7(v1)": FF(640, 960, 1280) + ["tile960"], "all7+tileF": FF(640, 960, 1280) + ["tile960", "tile2x2f"], "all+1600": FF(640, 960, 1280, 1600) + ["tile960", "tile2x2f"]}
TH = np.round(np.arange(0.05, 0.61, 0.025), 3)
for tgt in [c[:-6] for c in tl.columns if c.endswith("_count")]:
    y = tl[f"{tgt}_count"].values; rows = []
    for vs, keys in VS.items():
        for ct in (0.8, 0.7):
            for vote in sorted({1, 2, max(1, len(keys) // 2)}):
                sc = [merge(V[("train_images", f)], keys, tgt, 0.5, ct, vote) for f in tl.filename]
                E = np.array([np.abs(np.array([(s >= th).sum() for s in sc]) - y) for th in TH])
                sm = np.array([E[max(0, k - 1):k + 2].mean() for k in range(len(TH))]); k = int(np.argmin(sm))
                loo = np.mean([E[int(np.argmin(np.delete(E, i, 1).mean(1))), i] for i in range(len(y))])
                bias = (np.array([(s >= TH[k]).sum() for s in sc]) - y).mean()
                rows.append((loo, sm[k], E[k].mean(), TH[k], bias, vs, ct, vote))
    rows.sort(key=lambda r: (r[0] + r[1]) / 2); print(f"=== {p} {tgt}")
    for r in rows[:8]: print(f"   LOO {r[0]:.3f} | 고원 {r[1]:.3f} | MAE {r[2]:.3f} @thr {r[3]} | 편향 {r[4]:+.2f} | {r[5]} contain {r[6]} vote {r[7]}")
