"""잠정 답지와 제출·기준선 비교. 평가·진단 전용(13.3): 이 결과로 규칙을 다시 튜닝하지 않는다."""
import os, pickle, collections, sys
import numpy as np, pandas as pd
ANS = r"C:\Users\User\Downloads\submission (1).csv"
A = pd.read_csv(ANS); S = pd.read_csv("submission.csv"); B = pd.read_csv("versions/v1_pretrained_only_baseline.csv"); cols = ["airplane", "bird", "drone"]
print("== 0. 답지 형식 확인 ==")
print("  행 %d, 컬럼 %s, 파일명 일치 %s, 결측 %d, 이미지당 한 종류만 %s" % (len(A), list(A.columns), (A.filename.tolist() == S.filename.tolist()), A.isna().sum().sum(), bool(((A[cols] > 0).sum(axis=1) == 1).all())))
print("  답지 종류별 이미지: %s | 수량 합: %s" % ({k: int((A[k] > 0).sum()) for k in cols}, {k: int(A[k].sum()) for k in cols}))
kind = lambda D: D[cols].gt(0).values.argmax(1); names = np.array(cols)
ka, ks, kb = kind(A), kind(S), kind(B)
exact = lambda X: (X[cols].values == A[cols].values).all(1)
print("\n== 1. 이미지별 정확 일치율 (지문의 평가 지표) ==")
for nm, X in (("최종 제출(submission.csv)", S), ("사전학습만의 기준선", B)): e = exact(X); print("  %-26s %d / 300 = %.1f%%  | 종류만 맞힌 비율 %.1f%%" % (nm, e.sum(), 100 * e.mean(), 100 * (kind(X) == ka).mean()))
print("\n== 2. 종류 혼동행렬 (행 = 답지, 열 = 최종 제출) ==")
print(pd.crosstab(pd.Series(names[ka], name="답지"), pd.Series(names[ks], name="제출")).reindex(index=cols, columns=cols, fill_value=0).to_string())
print("  기준선:"); print(pd.crosstab(pd.Series(names[ka], name="답지"), pd.Series(names[kb], name="기준선")).reindex(index=cols, columns=cols, fill_value=0).to_string())
print("\n== 3. 종류별 성적 (최종) ==")
e = exact(S)
for i, k in enumerate(cols):
    m = ka == i; kc = (ks[m] == i); print("  답지 %-9s %3d장: 종류 맞힘 %3d (%.0f%%), 종류·수량 모두 맞힘 %3d (%.0f%%)" % (k, m.sum(), kc.sum(), 100 * kc.mean(), e[m].sum(), 100 * e[m].mean()))
print("\n== 4. 종류는 맞았는데 수량이 틀린 이미지 (최종) ==")
kc = ks == ka; cnt_wrong = kc & ~e; print("  %d장 (종류가 맞은 %d장 중 %.0f%%가 수량 오류)" % (cnt_wrong.sum(), kc.sum(), 100 * cnt_wrong.sum() / kc.sum()))
d = collections.Counter()
for i in np.where(cnt_wrong)[0]:
    k = cols[ka[i]]; d[(k, "과소" if S.loc[i, k] < A.loc[i, k] else "과다")] += 1
print("  종류·방향별:", dict(d))
for k in cols:
    m = (ka == names.tolist().index(k)) & cnt_wrong
    if m.any(): print("   %s: 답지 수량 대 제출 수량 =" % k, list(zip(A.loc[m, k], S.loc[m, k]))[:14])
print("\n== 5. 판정 규칙의 가지별 정확도 (노트북과 같은 규칙, 최초 학습 때의 예측으로 재현) ==")
D = pickle.load(open("_final_preds.pkl", "rb")); PD, PB = D["P_drone"], D["P_base"]
from PIL import Image
AIR, BIRD = 4, 14; TD, TP, TP_LO, AR_T = 0.35, 0.5, 0.1, 2.2
def ic(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = iw * ih; aa = (a[2] - a[0]) * (a[3] - a[1]); bb = (b[2] - b[0]) * (b[3] - b[1]); return i / (aa + bb - i + 1e-12), i / (min(aa, bb) + 1e-12)
def mb(bx, sc, thr, it=0.5, ct=0.8):
    keep = []
    for i in np.argsort(-sc):
        if sc[i] < thr: break
        if not any(ic(bx[i], bx[j])[0] >= it or ic(bx[i], bx[j])[1] >= ct for j in keep): keep.append(i)
    return keep
rows = []
for f in S.filename:
    c, s, b = PB[f]; dc, ds, db = PD[f]; W, H = Image.open(os.path.join("data/aerial_objects", f)).size
    sa = s[c == AIR].max() if (c == AIR).any() else 0.0; sb = s[c == BIRD].max() if (c == BIRD).any() else 0.0; sd = ds.max() if len(ds) else 0.0; ar = None
    if sb >= TP and sb >= sa: k, r = "bird", "base_bird"
    elif sa >= TP:
        i = np.where(c == AIR)[0][np.argmax(s[c == AIR])]; ar = (b[i][2] - b[i][0]) * W / max((b[i][3] - b[i][1]) * H, 1e-6)
        if ar >= AR_T: k, r = "airplane", "base_airplane"
        else: k, r = "drone", "base_drone_shape"
    elif sd >= TD: k, r = "drone", "drone_model"
    elif sb >= TP_LO and sb >= sa: k, r = "bird", "low_bird"
    elif sa >= TP_LO: k, r = "airplane", "low_airplane"
    else: k, r = "drone", "default"
    rows.append(dict(f=f, kind=k, why=r, sa=sa, sb=sb, sd=sd, ar=ar))
R = pd.DataFrame(rows); R["true"] = names[ka]; R["ok"] = R.kind == R.true
print("  재현한 규칙과 제출의 종류 일치: %d / 300" % (R.kind.values == names[ks]).sum())
print("  %-18s %-6s %-6s %s" % ("가지", "이미지", "종류 맞힘", "답지 종류 분포"))
for w, g in R.groupby("why"): print("  %-18s %-6d %-9s %s" % (w, len(g), "%d (%.0f%%)" % (g.ok.sum(), 100 * g.ok.mean()), dict(g.true.value_counts())))
R.to_csv("_answer_diag.csv", index=False)
print("\n== 6. 종류를 틀린 이미지 (제출 기준) ==")
w = np.where(ks != ka)[0]; print("  %d장:" % len(w), " ".join("%s(답:%s→제:%s)" % (S.filename[i][4:7], cols[ka[i]][:2], cols[ks[i]][:2]) for i in w))
