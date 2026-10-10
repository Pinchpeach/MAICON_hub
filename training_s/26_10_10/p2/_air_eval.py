"""비행기 과소 계수 방안 평가. 후보와 값은 평가 전에 고정(채택 기준: 정확 일치 +5장 이상, 나빠진 수가 좋아진 수의 1/3 이하). 잠정 답지로 1회 평가 — 낙관적."""
import pickle, sys, numpy as np, pandas as pd
AIR = 4; cols = ["airplane", "bird", "drone"]
S = pd.read_csv("submission.csv").set_index("filename"); A = pd.read_csv(r"C:\Users\User\Downloads\submission (1).csv").set_index("filename")
PB = pickle.load(open("_final_preds.pkl", "rb"))["P_base"]; PT = pickle.load(open("_air_tile.pkl", "rb")); files = sorted(PT)
area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
def rel(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = iw * ih; aa, bb = area(a), area(b)
    return i / (aa + bb - i + 1e-12), i / (min(aa, bb) + 1e-12), min(aa, bb) / (max(aa, bb) + 1e-12)
def base(f, thr=0.25):
    c, s, b = PB[f]; bx, sc = b[c == AIR], s[c == AIR]; keep = []
    for i in np.argsort(-sc):
        if sc[i] < thr: break
        if not any(rel(bx[i], bx[j])[0] >= 0.5 or rel(bx[i], bx[j])[1] >= 0.8 for j in keep): keep.append(i)
    return [bx[i] for i in keep]
def extra(f, src, conf=0.4, amax=0.05):
    c, s, b, e = PT[f][src]; m = (c == AIR) & (s >= conf) & (~e); bx, sc = b[m], s[m]; return [bx[i] for i in np.argsort(-sc) if area(bx[i]) < amax]
def add(kept, cand):
    kept = list(kept)
    for q in cand:
        if not any(rel(q, k)[0] >= 0.3 or (rel(q, k)[1] >= 0.8 and rel(q, k)[2] >= 0.3) for k in kept): kept.append(q)
    return kept
def n_low(f):                                                    # 640 에서 점수 0.1 이상인 서로 다른 비행기 후보 수(작은 박스는 큰 박스 안에 있어도 따로 센다)
    c, s, b = PB[f]; bx, sc = b[c == AIR], s[c == AIR]; keep = []
    for i in np.argsort(-sc):
        if sc[i] < 0.1: break
        if not any(rel(bx[i], bx[j])[0] >= 0.5 or (rel(bx[i], bx[j])[1] >= 0.8 and rel(bx[i], bx[j])[2] >= 0.1) for j in keep): keep.append(i)
    return len(keep)
def both(f, a, b_):
    return [q for q in extra(f, a) if any(rel(q, r)[0] >= 0.5 for r in extra(f, b_, conf=0.25))]
def tta(f):
    c, s, b, _ = PT[f]["tta640"]; bx, sc = b[c == AIR], s[c == AIR]; keep = []
    for i in np.argsort(-sc):
        if sc[i] < 0.25: break
        if not any(rel(bx[i], bx[j])[0] >= 0.5 or rel(bx[i], bx[j])[1] >= 0.8 for j in keep): keep.append(i)
    return max(len(keep), 1)
CAND = [("v3 (현재)", lambda f: len(base(f))),
        ("C1: 2x2 타일의 작은 박스 추가", lambda f: len(add(base(f), extra(f, "tile2")))),
        ("C2: 3x3 타일의 작은 박스 추가", lambda f: len(add(base(f), extra(f, "tile3")))),
        ("C3: 2x2 타일과 960 이 모두 본 박스만 추가", lambda f: len(add(base(f), both(f, "tile2", "s960")))),
        ("C4: 640 후보가 2개 이상일 때만 C1", lambda f: len(add(base(f), extra(f, "tile2"))) if n_low(f) >= 2 else len(base(f))),
        ("C5: 960 의 작은 박스 추가", lambda f: len(add(base(f), extra(f, "s960")))),
        ("C6: 640 TTA 수량으로 교체", tta),
        ("C7: 640 후보가 2개 이상일 때만 C3", lambda f: len(add(base(f), both(f, "tile2", "s960"))) if n_low(f) >= 2 else len(base(f)))]
ev = [f for f in files if A.loc[f, "airplane"] > 0 and (A.loc[f, cols].values > 0).sum() == 1]; t = np.array([int(A.loc[f, "airplane"]) for f in ev]); multi = t >= 2
print("비행기로 판정했고 답지도 비행기인 %d장 (정답 1대 %d장, 2대 이상 %d장)" % (len(ev), (~multi).sum(), multi.sum()))
print("%-38s %-8s %-8s %-10s %-10s %-9s %s" % ("방안", "정확", "v3 대비", "1대(정확)", "2대+(정확)", "평균오차", "좋아짐/나빠짐 | 과소/과다"))
P = {}
for n, fn in CAND:
    p = np.array([max(fn(f), 1) for f in ev]); P[n] = p; b0 = P["v3 (현재)"]; ok, ok0 = p == t, b0 == t
    print("%-38s %3d/%d  %+3d      %2d/%d      %2d/%d      %.2f      %d / %d | %d / %d" % (n, ok.sum(), len(ev), ok.sum() - ok0.sum(), (ok & ~multi).sum(), (~multi).sum(), (ok & multi).sum(), multi.sum(), np.abs(p - t).mean(), (ok & ~ok0).sum(), (~ok & ok0).sum(), (p < t).sum(), (p > t).sum()))
if len(sys.argv) > 1:
    D = pd.DataFrame({"f": ev, "true": t, **{n.split(":")[0]: P[n] for n, _ in CAND}}); D = D[(D.iloc[:, 2:].values != D[["true"]].values).any(1)]; print(D.to_string(index=False))
pickle.dump((ev, t, P), open("_air_eval.pkl", "wb"))
