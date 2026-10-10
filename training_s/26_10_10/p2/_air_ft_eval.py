"""비행기 전이학습 모델의 수량 평가. 임계값은 합성 검증(떼어 둔 기증 이미지)에서 먼저 정하고, 잠정 답지로는 그 값으로 1회만 평가한다(낙관적). 채택 기준: +5장 이상, 나빠짐 <= 좋아짐/3."""
import pickle, numpy as np, pandas as pd
cols = ["airplane", "bird", "drone"]; D = pickle.load(open("_air_ft.pkl", "rb")); VAL, PV, PA = D["VAL"], D["PV"], D["PA"]
area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = iw * ih; return i / (area(a) + area(b) - i + 1e-12)
def cnt(p, thr):
    sc, bx = p; keep = []
    for i in np.argsort(-sc):
        if sc[i] < thr: break
        if not any(iou(bx[i], bx[j]) >= 0.5 for j in keep): keep.append(i)
    return max(len(keep), 1)
TH = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6); vn = sorted(VAL); tv = np.array([VAL[n] for n in vn])
acc = [float((np.array([cnt(PV[n], th) for n in vn]) == tv).mean()) for th in TH]; print("합성 검증 70장 수량 정확도:", dict(zip(TH, np.round(acc, 3))))
best = [th for th, a in zip(TH, acc) if a >= max(acc) - 1e-9]; thr = best[len(best) // 2]; print("고른 임계값(최고 구간의 가운데):", thr)
S = pd.read_csv("submission.csv").set_index("filename"); A = pd.read_csv(r"C:\Users\User\Downloads\submission (1).csv").set_index("filename")
ev = [f for f in sorted(PA) if A.loc[f, "airplane"] > 0 and (A.loc[f, cols].values > 0).sum() == 1]; t = np.array([int(A.loc[f, "airplane"]) for f in ev]); v3 = np.array([int(S.loc[f, "airplane"]) for f in ev]); multi = t >= 2
m = np.array([cnt(PA[f], thr) for f in ev]); don = np.array([f in D["donors"] for f in ev])
print("%-34s %-7s %-7s %-7s %-7s %-8s %s" % ("방안", "정확", "v3 대비", "1대", "2대+", "평균오차", "좋아짐/나빠짐 | 과소/과다"))
for n, p in (("v3 (현재)", v3), ("A1: 비행기 모델 수량", m), ("A2: max(v3, 비행기 모델)", np.maximum(v3, m))):
    ok, ok0 = p == t, v3 == t; print("%-34s %2d/%d  %+3d     %2d/%d   %2d/%d   %.2f     %d / %d | %d / %d" % (n, ok.sum(), len(ev), ok.sum() - ok0.sum(), (ok & ~multi).sum(), (~multi).sum(), (ok & multi).sum(), multi.sum(), np.abs(p - t).mean(), (ok & ~ok0).sum(), (~ok & ok0).sum(), (p < t).sum(), (p > t).sum()))
print("기증 이미지였던 %d장(학습에 '한 대'로 들어감): 정답 2대 이상 %d장, 그중 A1 정확 %d장" % (don.sum(), (don & multi).sum(), (don & multi & (m == t)).sum()))
X = pd.DataFrame(dict(f=ev, true=t, v3=v3, A1=m, donor=don.astype(int))); print(X[(X.v3 != X.true) | (X.A1 != X.true)].to_string(index=False))
