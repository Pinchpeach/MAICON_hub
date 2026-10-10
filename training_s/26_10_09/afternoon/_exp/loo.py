"""설정 묶음(관점·NMS·포함도·투표)은 고정하고 임계값만 폴드 안에서 고르는 LOO. 임계값 고원(±0.025 평균)도 함께 본다."""
import sys, pickle, numpy as np, collections
p, tgt = sys.argv[1], sys.argv[2]; names, P, y = pickle.load(open(f"_exp/{p}_{tgt}_search.pkl", "rb"))
fam = collections.defaultdict(list)
for i, n in enumerate(names): fam[n[:4]].append(i)
rows = []
for f, idx in fam.items():
    E = np.abs(P[idx] - y); th = [names[i][4] for i in idx]
    sm = np.array([E[max(0, k - 1):k + 2].mean() for k in range(len(idx))])     # 임계값 이웃 평균(고원)
    loo = [E[int(np.argmin(np.delete(E, i, 1).mean(1))), i] for i in range(len(y))]
    k = int(np.argmin(sm)); rows.append((np.mean(loo), sm[k], E[k].mean(), th[k], (P[idx][k] - y).mean(), f))
rows.sort(key=lambda r: (r[0] + r[1]) / 2)
print(f"=== {p} {tgt}: LOO(임계값만 선택) / 고원 평균 / 그 지점 MAE / thr / 편향")
for r in rows[:10]: print(f"   LOO {r[0]:.3f} | 고원 {r[1]:.3f} | MAE {r[2]:.3f} @thr {r[3]} | 편향 {r[4]:+.2f} | {r[5]}")
base = [r for r in rows if r[5] in (("f640", 0.5, 1.01, 1), ("f1280", 0.5, 1.01, 1))]
for r in base: print(f"   (참고) LOO {r[0]:.3f} | 고원 {r[1]:.3f} | MAE {r[2]:.3f} @thr {r[3]} | {r[5]}")
