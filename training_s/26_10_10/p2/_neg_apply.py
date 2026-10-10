"""부정 예시로 학습한 드론 모델(겹 0 가중치)로 v3 종류 판정 규칙을 테스트 300장에 한 번 적용하고 잠정 답지와 비교한다(평가 전용)."""
import os, pickle, numpy as np, pandas as pd, torch
from ultralytics import YOLO
AIR, BIRD = 4, 14; TAU, TD, TP, TP_LO = 0.4, 0.35, 0.5, 0.1; AO = "data/aerial_objects"; dev = 0 if torch.cuda.is_available() else "cpu"
W = pickle.load(open("_neg_eval.pkl", "rb"))["W"]; PB = pickle.load(open("_final_preds.pkl", "rb"))["P_base"]; files = sorted(PB)
fn = "_neg_test_pred.pkl"
if os.path.exists(fn): PD = pickle.load(open(fn, "rb"))
else:
    m = YOLO(W); PD = {}
    for i in range(0, len(files), 10):
        for f, r in zip(files[i:i + 10], m.predict([os.path.join(AO, x) for x in files[i:i + 10]], imgsz=640, conf=0.05, iou=0.6, device=dev, verbose=False)): PD[f] = (r.boxes.conf.cpu().numpy(), r.boxes.xyxyn.cpu().numpy())
    pickle.dump(PD, open(fn, "wb"))
def ic(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = iw * ih; aa = (a[2] - a[0]) * (a[3] - a[1]); bb = (b[2] - b[0]) * (b[3] - b[1]); return i / (aa + bb - i + 1e-12), i / (min(aa, bb) + 1e-12)
def mb(bx, sc, thr, it=0.5, ct=0.8):
    keep = []
    for i in np.argsort(-sc):
        if sc[i] < thr: break
        if not any(ic(bx[i], bx[j])[0] >= it or ic(bx[i], bx[j])[1] >= ct for j in keep): keep.append(i)
    return keep
cols = ["airplane", "bird", "drone"]; A = pd.read_csv(r"C:\Users\User\Downloads\submission (1).csv"); V2 = pd.read_csv("submission.csv"); rows = []; why = []
for f in files:
    c, s, b = PB[f]; ds, db = PD[f]
    sa = s[c == AIR].max() if (c == AIR).any() else 0.0; sb = s[c == BIRD].max() if (c == BIRD).any() else 0.0; sd = ds.max() if len(ds) else 0.0
    if sd >= TAU: k, r = "drone", "drone_model"
    elif sb >= TP and sb >= sa: k, r = "bird", "base_bird"
    elif sa >= TP: k, r = "airplane", "base_airplane"
    elif sd >= TD: k, r = "drone", "drone_low"
    elif sb >= TP_LO and sb >= sa: k, r = "bird", "low_bird"
    elif sa >= TP_LO: k, r = "airplane", "low_airplane"
    else: k, r = "drone", "default"
    if k == "airplane": n = len(mb(b[c == AIR], s[c == AIR], 0.25))
    elif k == "bird": n = len(mb(b[c == BIRD], s[c == BIRD], 0.25))
    else: n = len(mb(db, ds, TD, 0.5, 1.01))
    n = max(n, 1); rows.append(dict(filename=f, **{x: (n if k == x else 0) for x in cols})); why.append(r)
S = pd.DataFrame(rows); S.to_csv("_v3_neg_submission.csv", index=False); names = np.array(cols)
kind = lambda D: D[cols].gt(0).values.argmax(1); ka = kind(A); e = lambda X: (X[cols].values == A[cols].values).all(1)
print("== v3(부정 예시 학습) 대 v2 (잠정 답지, 평가 전용) ==")
for n, X in (("v2", V2), ("v3", S)): print("  %-4s 정확 일치 %d/300 = %.1f%% | 종류 맞힘 %.1f%%" % (n, e(X).sum(), 100 * e(X).mean(), 100 * (kind(X) == ka).mean()))
print("  종류 혼동행렬 v3 (행 = 답지):"); print(pd.crosstab(pd.Series(names[ka], name="답지"), pd.Series(names[kind(S)], name="v3")).reindex(index=cols, columns=cols, fill_value=0).to_string())
ch = (S[cols].values != V2[cols].values).any(1); ok3, ok2 = e(S), e(V2)
print("  v2 에서 바뀐 이미지 %d장: 좋아짐 %d / 나빠짐 %d" % (ch.sum(), (ok3 & ~ok2).sum(), (~ok3 & ok2).sum()))
Wy = pd.Series(why); R = pd.DataFrame(dict(why=Wy, ok=(kind(S) == ka), true=names[ka]))
print("  가지별 종류 정확도:"); [print("   %-16s %3d장 %s 맞힘 %d (%.0f%%)" % (w, len(g), "", g.ok.sum(), 100 * g.ok.mean()), dict(g.true.value_counts())) for w, g in R.groupby("why")]
print("  이미지 수(v3): %s | 답지: %s" % ({k: int((S[k] > 0).sum()) for k in cols}, {k: int((A[k] > 0).sum()) for k in cols}))
