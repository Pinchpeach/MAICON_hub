import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "_exp")
from lib import merge
TH = np.round(np.arange(0.05, 0.61, 0.025), 3)
FAM = {("problem02", "person"): (["full960", "full1280", "tile960"], 0.5, 0.7, 2), ("problem04", "person"): (["full640", "flip640", "full960", "flip960", "full1280", "flip1280"], 0.5, 0.8, 1),
       ("problem04", "vehicle"): (["full640", "full960", "full1280"], 0.5, 0.7, 2)}
def score(S, y):
    E = np.array([np.abs(np.array([(s >= th).sum() for s in S]) - y) for th in TH]); sm = np.array([E[max(0, k - 1):k + 2].mean() for k in range(len(TH))]); k = int(np.argmin(sm))
    loo = np.mean([E[int(np.argmin(np.delete(E, i, 1).mean(1))), i] for i in range(len(y))]); pr = np.array([(s >= TH[k]).sum() for s in S])
    return E[k].mean(), loo, TH[k], (pr - y).mean(), pr
for p in ("problem02", "problem04"):
    V, stats = pickle.load(open(f"_exp/{p}_bright.pkl", "rb")); tl = pd.read_csv(f"{p}/data/train_labels.csv"); st = np.array(stats)
    for tgt in [c[:-6] for c in tl.columns if c.endswith("_count")]:
        y = tl[f"{tgt}_count"].values; keys, nms, ct, vote = FAM[(p, tgt)]
        S0 = [merge(V["orig"][f], keys, tgt, nms, ct, vote) for f in tl.filename]; mae0, loo0, th0, b0, pr0 = score(S0, y); err = pr0 - y
        print(f"=== {p} {tgt} | 기준(원본) MAE {mae0:.3f} LOO {loo0:.3f} thr {th0} 편향 {b0:+.2f}")
        print("   오류와 이미지 통계의 상관(스피어만):", {n: round(float(pd.Series(v).corr(pd.Series(err), method='spearman')), 2) for n, v in zip(["밝기 평균", "대비(표준편차)", "어두운 화소 비율", "선명도"], st.T)},
              "| |오류|와의 상관:", {n: round(float(pd.Series(v).corr(pd.Series(np.abs(err)), method='spearman')), 2) for n, v in zip(["밝기", "대비", "어두움", "선명도"], st.T)})
        dark = st[:, 0] < np.median(st[:, 0]); print(f"   어두운 절반 MAE {np.abs(err[dark]).mean():.2f} (편향 {err[dark].mean():+.2f}) | 밝은 절반 MAE {np.abs(err[~dark]).mean():.2f} (편향 {err[~dark].mean():+.2f})")
        for k in ("autocontrast", "gamma", "clahe", "equalize", "sharpen"):
            S1 = [merge(V[k][f], keys, tgt, nms, ct, vote) for f in tl.filename]; a = score(S1, y)
            # 원본 + 전처리본을 함께(관점 2배): 관점 이름이 겹치지 않게 합친다
            S2 = []
            for f in tl.filename:
                vv = dict(V["orig"][f]); vv.update({"p_" + n: V[k][f][n] for n in keys}); S2.append(merge(vv, keys + ["p_" + n for n in keys], tgt, nms, ct, vote))
            b = score(S2, y)
            S3 = []
            for f in tl.filename:
                vv = dict(V["orig"][f]); vv.update({"p_" + n: V[k][f][n] for n in keys}); S3.append(merge(vv, keys + ["p_" + n for n in keys], tgt, nms, ct, max(vote, len(keys))))
            c = score(S3, y)
            print(f"   {k:13s} 대체: MAE {a[0]:.3f} LOO {a[1]:.3f} 편향 {a[3]:+.2f} | 원본과 함께: MAE {b[0]:.3f} LOO {b[1]:.3f} | 함께+절반 이상 동의: MAE {c[0]:.3f} LOO {c[1]:.3f}")
        # 밝기 조건부 임계값: 어두운 절반/밝은 절반에 임계값 따로 (LOO)
        E = np.array([np.abs(np.array([(s >= th).sum() for s in S0]) - y) for th in TH]); loo = []
        for i in range(len(y)):
            tr = np.delete(np.arange(len(y)), i); med = np.median(st[tr, 0]); grp = st[:, 0] < med
            sel = tr[grp[tr] == grp[i]]; loo.append(E[int(np.argmin(E[:, sel].mean(1))), i])
        print(f"   밝기별 임계값 2개(LOO): {np.mean(loo):.3f}  (기준 LOO {loo0:.3f})")
