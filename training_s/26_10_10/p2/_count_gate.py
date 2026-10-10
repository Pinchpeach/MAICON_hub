import pickle, numpy as np, pandas as pd
exec(open("_count_ceiling.py", encoding="utf-8").read().split("TH = (0.05")[0])
fs = [f for f in files if KIND[f] == "airplane" and A.loc[f, "airplane"] > 0 and (A.loc[f, cols].values > 0).sum() == 1]; t = np.array([int(A.loc[f, "airplane"]) for f in fs])
c = lambda P, f, th, mr: cnt(P[f][2][P[f][0] == AIR], P[f][1][P[f][0] == AIR], th, mr)
n640 = np.array([c(PB, f, 0.25, 0.0) for f in fs]); print("정답 2대 이상 28장 중 640 이 2대 이상으로 본 이미지:", int(((n640 >= 2) & (t >= 2)).sum()), "| 정답 1대 69장 중 640 이 2대 이상으로 본 이미지:", int(((n640 >= 2) & (t == 1)).sum()))
print("현재(640, 0.25): %d / %d" % ((n640 == t).sum(), len(t)))
for th in (0.25, 0.3, 0.4, 0.5):
    for mr in (0.0, 0.1):
        n12 = np.array([c(P12, f, th, mr) for f in fs]); g = np.where(n640 >= 2, np.maximum(n12, n640), n640); g2 = np.where(n640 >= 2, n12, n640)
        print("  640 이 2대 이상일 때만 1280 사용 (임계값 %.2f, 면적비 %s): 1280 값 %d / max(640,1280) %d" % (th, "있음" if mr else "없음", (g2 == t).sum(), (g == t).sum()))
d = n640 - t; print("640 수량 - 정답 분포(정답 2대 이상):", pd.Series(d[t >= 2]).value_counts().sort_index().to_dict())
