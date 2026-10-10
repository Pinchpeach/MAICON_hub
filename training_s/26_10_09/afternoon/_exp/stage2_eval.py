"""2차 검증 규칙 평가: 최종 수량 = 1차로 센 수 + (탈락 후보 중 조건을 만족하는 수). 조건의 임계값은 LOO 로 고른다."""
import sys, pickle, itertools, numpy as np, pandas as pd
p, tgt = sys.argv[1], sys.argv[2]; D = pickle.load(open(f"_exp/{p}_{tgt}_stage2.pkl", "rb")); tl = pd.read_csv(f"{p}/data/train_labels.csv")
y = tl[f"{tgt}_count"].values; T = D[D.split == "train_images"].copy(); T["i"] = T.f.map({f: i for i, f in enumerate(tl.filename)})
base = T[T.counted].groupby("i").size().reindex(range(len(y))).fillna(0).values; C = T[~T.counted].copy(); C["zmax"] = C[["z3", "z6"]].max(1); C["zmean"] = C[["z3", "z6"]].mean(1)
print(f"=== {p} {tgt} | 1차만: MAE {np.abs(base - y).mean():.3f} 편향 {(base - y).mean():+.2f} | 탈락 후보 {len(C)}개")
print("   탈락 후보의 확대 재검출 conf 분포: z3 중앙값 %.2f, z6 중앙값 %.2f | z6>=0.3 %d개, >=0.5 %d개" % (C.z3.median(), C.z6.median(), (C.z6 >= 0.3).sum(), (C.z6 >= 0.5).sum()))
def loo(E): return np.mean([E[int(np.argmin(np.delete(E, i, 1).mean(1))), i] for i in range(len(y))])
def cnt(mask): return C[mask].groupby("i").size().reindex(range(len(y))).fillna(0).values
ZT = np.round(np.arange(0.10, 0.81, 0.05), 2); res = []
for zname in ("z3", "z6", "zmax", "zmean"):
    for ov in (1.01, 0.3):                     # 이미 센 객체와 겹치는 후보 제외 여부
        for mc in (0.03, 0.06, 0.10):           # 1차 최고 conf 하한
            P = np.array([base + cnt((C[zname] >= zt) & (C.ov_acc < ov) & (C.maxconf >= mc)) for zt in ZT]); E = np.abs(P - y); k = int(np.argmin(E.mean(1)))
            res.append((loo(E), E[k].mean(), ZT[k], (P[k] - y).mean(), zname, ov, mc))
res.sort()
for r in res[:8]: print(f"   LOO {r[0]:.3f} | MAE {r[1]:.3f} @확대 conf>={r[2]} | 편향 {r[3]:+.2f} | {r[4]}, 겹침제외<{r[5]}, 1차 conf>={r[6]}")
pickle.dump(res, open(f"_exp/{p}_{tgt}_stage2_res.pkl", "wb"))
