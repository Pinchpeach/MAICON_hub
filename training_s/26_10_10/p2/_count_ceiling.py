"""진단: 기본 yolov8s 의 박스 후보로 비행기·새 수량이 정답과 같아질 수 있는 상한. 임계값·해상도·병합 규칙을 훑는다(답지로 훑으므로 선택에 쓰면 낙관적)."""
import pickle, numpy as np, pandas as pd
AIR, BIRD = 4, 14; cols = ["airplane", "bird", "drone"]
S = pd.read_csv("submission.csv").set_index("filename"); A = pd.read_csv(r"C:\Users\User\Downloads\submission (1).csv").set_index("filename"); files = list(S.index)
PB = pickle.load(open("_final_preds.pkl", "rb"))["P_base"]; P12 = pickle.load(open("_base1280.pkl", "rb"))
KIND = {f: cols[int(np.argmax(S.loc[f, cols].values))] for f in files}; area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
def rel(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = iw * ih; aa, bb = area(a), area(b); return i / (aa + bb - i + 1e-12), i / (min(aa, bb) + 1e-12), min(aa, bb) / (max(aa, bb) + 1e-12)
def cnt(bx, sc, thr, mr, it=0.5):
    keep = []
    for i in np.argsort(-sc):
        if sc[i] < thr: break
        if not any(rel(bx[i], bx[j])[0] >= it or (rel(bx[i], bx[j])[1] >= 0.8 and rel(bx[i], bx[j])[2] >= mr) for j in keep): keep.append(i)
    return max(len(keep), 1)
TH = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
for kind, cid in (("airplane", AIR), ("bird", BIRD)):
    fs = [f for f in files if KIND[f] == kind and A.loc[f, kind] > 0 and (A.loc[f, cols].values > 0).sum() == 1]; t = np.array([int(A.loc[f, kind]) for f in fs]); multi = t >= 2
    print("== %s: 종류가 맞은 %d장 (정답 2개 이상 %d장) — 수량이 정확히 맞는 이미지 수 ==" % (kind, len(fs), multi.sum()))
    print("  %-22s " % "설정" + " ".join("%5s" % ("%.2f" % x) for x in TH) + "   (열 = 점수 임계값)")
    best = np.zeros(len(fs), bool)
    for nm, P in (("640", PB), ("1280", P12)):
        for mr in (0.0, 0.1):
            row = []
            for th in TH:
                p = np.array([cnt(P[f][2][P[f][0] == cid], P[f][1][P[f][0] == cid], th, mr) for f in fs]); row.append((p == t).sum()); best |= (p == t)
            print("  %-22s " % ("%s, 면적비 조건 %s" % (nm, "있음" if mr else "없음")) + " ".join("%5d" % x for x in row))
    # 정답 2개 이상에서만
    for nm, P in (("640", PB), ("1280", P12)):
        row = [(np.array([cnt(P[f][2][P[f][0] == cid], P[f][1][P[f][0] == cid], th, 0.1) for f in fs])[multi] == t[multi]).sum() for th in TH]
        print("  %-22s " % ("%s, 정답 2개 이상만" % nm) + " ".join("%5d" % x for x in row))
    print("  어떤 설정에서든 한 번이라도 맞은 이미지(이미지마다 최선의 설정을 고른 비현실적 상한): %d / %d | 그중 정답 2개 이상 %d / %d" % (best.sum(), len(fs), (best & multi).sum(), multi.sum()))
    # 정답 대비 후보 수: 임계값 0.05 에서도 정답보다 적게 잡히는 이미지
    for nm, P in (("640", PB), ("1280", P12)):
        low = np.array([cnt(P[f][2][P[f][0] == cid], P[f][1][P[f][0] == cid], 0.05, 0.1) for f in fs]); print("  %s 에서 임계값 0.05 까지 낮춰도 정답 수량에 못 미치는 이미지: %d장 (정답 2개 이상 %d장 중 %d장)" % (nm, (low < t).sum(), multi.sum(), ((low < t) & multi).sum()))
