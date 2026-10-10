"""개수 세기 개선 후보 R(면적 비율 조건), H(1280 의 작은 박스 추가), D(드론 2번째 박스부터 0.5) 를 v3 의 종류 판정 위에서 1회 평가한다. 정답 공개 후 평가라 낙관적."""
import pickle, numpy as np, pandas as pd
AIR, BIRD = 4, 14; cols = ["airplane", "bird", "drone"]; TD = 0.35
S = pd.read_csv("submission.csv").set_index("filename"); A = pd.read_csv(r"C:\Users\User\Downloads\submission (1).csv").set_index("filename"); files = list(S.index)
PB = pickle.load(open("_final_preds.pkl", "rb"))["P_base"]; P12 = pickle.load(open("_base1280.pkl", "rb")); PD = pickle.load(open("_v3_drone_pred.pkl", "rb"))
KIND = {f: cols[int(np.argmax(S.loc[f, cols].values))] for f in files}
area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
def rel(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = iw * ih; aa, bb = area(a), area(b)
    return i / (aa + bb - i + 1e-12), i / (min(aa, bb) + 1e-12), min(aa, bb) / (max(aa, bb) + 1e-12)
def merge(bx, sc, thr, it=0.5, ct=0.8, min_ratio=0.0):
    """IoU >= it 이면 같은 객체. 포함도 >= ct 이면 같은 객체이되, min_ratio > 0 이면 면적 비율이 그보다 작을 때는 다른 객체로 둔다."""
    keep = []
    for i in np.argsort(-sc):
        if sc[i] < thr: break
        dup = False
        for j in keep:
            io, co, ra = rel(bx[i], bx[j])
            if io >= it or (co >= ct and ra >= min_ratio): dup = True; break
        if not dup: keep.append(i)
    return [bx[i] for i in keep]
def count(f, R=False, H=False, D=False, bird_too=True):
    k = KIND[f]
    if k == "drone":
        ds, db = PD[f]; o = np.argsort(-ds); n = 0; kept = []
        for i in o:
            if ds[i] < TD: break
            if D and kept and ds[i] < 0.5: continue                      # D: 두 번째 박스부터는 0.5 이상만
            if any(rel(db[i], q)[0] >= 0.5 for q in kept): continue
            kept.append(db[i])
        return max(len(kept), 1)
    c, s, b = PB[f]; cid = AIR if k == "airplane" else BIRD; use = (k == "airplane") or bird_too
    kept = merge(b[c == cid], s[c == cid], 0.25, min_ratio=0.1 if (R and use) else 0.0)
    if H and use:
        c2, s2, b2 = P12[f]; m = (c2 == cid) & (s2 >= 0.25)
        for i in np.where(m)[0][np.argsort(-s2[m])]:
            if area(b2[i]) >= 0.02: continue                                # H: 작은 박스만 추가
            if any(rel(b2[i], q)[0] >= 0.3 or (rel(b2[i], q)[1] >= 0.8 and rel(b2[i], q)[2] >= 0.3) for q in kept): continue
            kept.append(b2[i])
    return max(len(kept), 1)
def table(**kw):
    rows = [dict(filename=f, **{x: (count(f, **kw) if KIND[f] == x else 0) for x in cols}) for f in files]; return pd.DataFrame(rows).set_index("filename")
V3 = table(); print("v3 수량 재현: %d / 300" % (V3[cols].values == S[cols].values).all(1).sum())
ok = lambda X: (X[cols].values == A[cols].values).all(1); base = ok(V3)
print("%-34s %-16s %-8s %s" % ("방안", "정확 일치", "v3 대비", "바뀐 이미지: 좋아짐 / 나빠짐"))
CAND = [("v3 (현재)", {}), ("R: 면적 비율 조건 (비행기·새)", dict(R=True)), ("R: 비행기에만", dict(R=True, bird_too=False)), ("H: 1280 의 작은 박스 추가", dict(H=True)), ("R + H", dict(R=True, H=True)),
        ("R + H (비행기에만)", dict(R=True, H=True, bird_too=False)), ("D: 드론 2번째부터 0.5", dict(D=True)), ("R + H + D", dict(R=True, H=True, D=True)), ("R + H(비행기만) + D", dict(R=True, H=True, D=True, bird_too=False))]
out = {}
for n, kw in CAND:
    X = table(**kw); o = ok(X); ch = (X[cols].values != V3[cols].values).any(1); out[n] = X
    print("%-34s %3d/300 (%.1f%%) %+3d   %2d장: %d / %d" % (n, o.sum(), 100 * o.mean(), o.sum() - base.sum(), ch.sum(), (o & ~base).sum(), (~o & base).sum()))
for n in ("R + H + D",):
    X = out[n]; print("\n[%s] 종류별 수량 정확도 (종류가 맞은 이미지 기준):" % n)
    for k in cols:
        m = np.array([KIND[f] == k and (A.loc[f, cols].values > 0).sum() == 1 and A.loc[f, k] > 0 for f in files]); t = A[k].values[m]; p3 = V3[k].values[m]; px = X[k].values[m]
        print("   %-9s v3 %d/%d -> %d/%d | 정답 2개 이상: v3 %d/%d -> %d/%d" % (k, (p3 == t).sum(), m.sum(), (px == t).sum(), m.sum(), ((p3 == t) & (t >= 2)).sum(), (t >= 2).sum(), ((px == t) & (t >= 2)).sum(), (t >= 2).sum()))
pickle.dump(out, open("_count_candidates.pkl", "wb"))
