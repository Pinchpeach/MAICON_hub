import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, LeaveOneOut
t = pd.read_csv("problem05/data/train.csv"); te = pd.read_csv("problem05/data/test.csv"); F = ["pushup", "run_time", "sleep_quality", "stress_level"]; X = t[F].values.astype(float); fit = (t.status == "fit").values; rec = t.recovery_days.values.astype(float)
SIGN = np.array([1, -1, 1, -1]); SC = 2.46475151; V = X * SIGN; Vt = te[F].values * SIGN
th = lambda Xa, ya: (Xa * SIGN)[ya].min(0); box = lambda Xa, ya, Xb: ((Xb * SIGN) >= th(Xa, ya)).all(1)
lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000))
T = th(X, fit)
print("== 1. 규칙이 몇 명에게 기대고 있는가 ==")
for j, c in enumerate(F):
    vals = np.sort(V[fit, j]); n_at = int((V[fit, j] == T[j]).sum()); print("  %-14s 임계값 %5.1f 를 정하는 fit %d명 | 그 사람을 빼면 임계값 %5.1f 로 이동 (차이 %.1f)" % (c, T[j] * SIGN[j], n_at, vals[n_at] * SIGN[j] if n_at < len(vals) else np.nan, abs(vals[n_at] - T[j])))
crit = [(V[fit, j] == T[j]) & (np.array([(V[fit, j] == T[j]).sum()] * fit.sum()) == 1) for j in range(4)]; sole = np.any(crit, axis=0)
print("  혼자서 임계값을 정하는 fit: %d명 / 10명 (이 사람이 학습에 없으면 그 사람은 unfit 으로 분류됨)" % sole.sum())
print("\n== 2. 한 명씩 빼고 검증 (Leave-One-Out, 80회) ==")
pb = np.zeros(80, bool); pl = np.zeros(80, bool)
for a, b in LeaveOneOut().split(X): pb[b] = box(X[a], fit[a], X[b]); pl[b] = lr().fit(X[a], fit[a]).predict(X[b])
for n, p in (("상자 규칙", pb), ("로지스틱", pl)): print("  %-8s 정확도 %.4f (%d/80) | 놓친 fit %d명, 잘못 fit 으로 본 unfit %d명" % (n, (p == fit).mean(), (p == fit).sum(), (fit & ~p).sum(), (~fit & p).sum()))
print("  두 모델이 서로 다르게 답한 행: %d (상자만 맞음 %d, 로지스틱만 맞음 %d)" % ((pb != pl).sum(), ((pb == fit) & (pl != fit)).sum(), ((pl == fit) & (pb != fit)).sum()))
print("\n== 3. 기준별 필요성: 한 기준을 빼면 unfit 몇 명이 fit 으로 잘못 들어오는가 ==")
for j, c in enumerate(F):
    p = (np.delete(V, j, 1) >= np.delete(T, j)).all(1); print("  %-14s 를 빼면 거짓 fit %d명 (정확도 %.4f)" % (c, (p & ~fit).sum(), (p == fit).mean()))
print("\n== 4. 재표집 안정성 (학습 80행을 2,000번 재표집) ==")
rng = np.random.default_rng(0); B = 2000; TH = np.zeros((B, 4)); oob_acc = []; tp = np.zeros((B, len(te)), bool)
for k in range(B):
    i = rng.integers(0, 80, 80)
    if fit[i].sum() < 2: TH[k] = np.nan; continue
    TH[k] = th(X[i], fit[i]); o = np.setdiff1d(np.arange(80), i); oob_acc.append((((V[o] >= TH[k]).all(1)) == fit[o]).mean()); tp[k] = (Vt >= TH[k]).all(1)
for j, c in enumerate(F): print("  %-14s 임계값이 전체 학습과 같았던 비율 %.2f | 5~95%% 범위 %.1f ~ %.1f" % (c, np.nanmean(TH[:, j] == T[j]), *np.sort(np.nanpercentile(TH[:, j] * SIGN[j], [5, 95]))))
print("  재표집에 뽑히지 않은 행으로 잰 정확도: 평균 %.4f (5~95%% %.3f ~ %.3f)" % (np.mean(oob_acc), *np.percentile(oob_acc, [5, 95])))
print("  테스트 예측이 재표집마다 바뀌는 행:", {te.soldier_id[i]: round(float(tp[:, i].mean()), 3) for i in range(len(te)) if 0 < tp[:, i].mean() < 1} or "없음", "| 항상 fit:", [te.soldier_id[i] for i in range(len(te)) if tp[:, i].mean() == 1])
print("\n== 5. 우연히 나온 규칙일 가능성: 라벨을 무작위로 섞었을 때 상자 규칙이 완벽히 가르는 비율 ==")
perfect = 0; NP = 5000
for _ in range(NP):
    y = rng.permutation(fit); perfect += ((V >= V[y].min(0)).all(1) == y).all()
print("  섞은 라벨 %d번 중 완벽 분리 %d번 (%.4f)" % (NP, perfect, perfect / NP))
print("\n== 6. 반복 교차검증 비교 (Stratified 5-Fold x 20) ==")
cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0).split(X, fit)); rows = {}
for name, pred in (("상자 규칙", box), ("로지스틱", lambda Xa, ya, Xb: lr().fit(Xa, ya).predict(Xb).astype(bool)), ("정답 status(하한)", None), ("평균 예측(status 무시)", "const")):
    acc, rm, fn, fp = [], [], [], []
    for r in range(20):
        ps = np.zeros(80, bool); oof = np.zeros(80)
        for a, b in cv[r * 5:(r + 1) * 5]:
            if pred == "const": ps[b] = False; oof[b] = rec[a].mean(); continue
            p = fit[b] if pred is None else pred(X[a], fit[a], X[b]); ps[b] = p; oof[b] = np.where(p, rec[a][fit[a]].mean(), rec[a][~fit[a]].mean())
        acc.append((ps == fit).mean()); rm.append(np.sqrt(np.mean((rec - oof) ** 2))); fn.append((fit & ~ps).sum()); fp.append((~fit & ps).sum())
    rows[name] = (np.array(acc), np.array(rm), np.mean(fn), np.mean(fp))
print("  %-22s %-8s %-10s %-10s %-9s %-9s %s" % ("모델", "정확도", "놓친 fit", "거짓 fit", "회복RMSE", "회복점수", "합계"))
for n, (acc, rm, fn, fp) in rows.items():
    rs = 50 / (1 + rm / SC); print("  %-22s %.4f   %-10.2f %-10.2f %.4f    %.2f     %.2f" % (n, acc.mean(), fn, fp, rm.mean(), rs.mean(), (50 * acc + rs).mean()))
d = rows["상자 규칙"][0] - rows["로지스틱"][0]; print("  정확도 차(상자 - 로지스틱): 평균 %+.4f, 20회 중 %d회 상자가 높음, %d회 같음" % (d.mean(), (d > 0).sum(), (d == 0).sum()))
d = rows["평균 예측(status 무시)"][1] - rows["상자 규칙"][1]; print("  회복 RMSE 차(평균 예측 - 2단계 상자): 평균 %+.4f, 20회 중 %d회 2단계가 낮음" % (d.mean(), (d > 0).sum()))
