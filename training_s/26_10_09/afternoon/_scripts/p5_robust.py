import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import trim_mean
from sklearn.model_selection import RepeatedStratifiedKFold
t = pd.read_csv("problem05/data/train.csv"); te = pd.read_csv("problem05/data/test.csv"); F = ["pushup", "run_time", "sleep_quality", "stress_level"]; X = t[F].values.astype(float); fit = (t.status == "fit").values; rec = t.recovery_days.values.astype(float); SIGN = np.array([1, -1, 1, -1]); SC = 2.46475151; R = 20
print("== 회복일수 분포 (특이값 확인) ==")
print("  fit   (10명):", sorted(rec[fit].astype(int).tolist()), "| 평균 %.2f 중앙값 %.1f 10%%절사평균 %.2f" % (rec[fit].mean(), np.median(rec[fit]), trim_mean(rec[fit], 0.1)))
print("  unfit (70명):", dict(pd.Series(rec[~fit].astype(int)).value_counts().sort_index()), "| 평균 %.2f 중앙값 %.1f 10%%절사평균 %.2f" % (rec[~fit].mean(), np.median(rec[~fit]), trim_mean(rec[~fit], 0.1)))
print("  unfit 에서 평균 ±2 표준편차 밖:", rec[~fit][np.abs(rec[~fit] - rec[~fit].mean()) > 2 * rec[~fit].std()].astype(int).tolist(), "| fit 에서:", rec[fit][np.abs(rec[fit] - rec[fit].mean()) > 2 * rec[fit].std()].astype(int).tolist())
EST = {"평균 (v1)": np.mean, "평균 반올림": lambda v: np.round(np.mean(v)), "중앙값": np.median, "10% 절사평균": lambda v: trim_mean(v, 0.1), "10% 절사평균 반올림": lambda v: np.round(trim_mean(v, 0.1)), "윈저화 평균(양끝 1개 안으로)": lambda v: np.mean(np.clip(v, np.sort(v)[1], np.sort(v)[-2]))}
cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=R, random_state=0).split(X, fit)); base = None
print("\n== 회복일수 예측값 추정 방식 비교 (status 는 상자 규칙, 5-Fold x 20) ==")
for n, est in EST.items():
    acc, rm, mae = [], [], []
    for r in range(R):
        ps = np.zeros(80, bool); oof = np.zeros(80)
        for a, b in cv[r * 5:(r + 1) * 5]:
            th = (X[a] * SIGN)[fit[a]].min(0); pb = ((X[b] * SIGN) >= th - 1e-9).all(1); ps[b] = pb; oof[b] = np.where(pb, est(rec[a][fit[a]]), est(rec[a][~fit[a]]))
        acc.append((ps == fit).mean()); rm.append(np.sqrt(np.mean((rec - oof) ** 2))); mae.append(np.abs(rec - oof).mean())
    acc, rm = np.array(acc), np.array(rm); sc = 50 * acc + 50 / (1 + rm / SC); base = sc if base is None else base
    print("  %-30s 회복 RMSE %.4f MAE %.3f  점수 %.3f  v1 대비 %+.3f (%2d/20)" % (n, rm.mean(), np.mean(mae), sc.mean(), (sc - base).mean(), int((sc > base).sum())))
print("\n== 전체 학습에서 정한 예측값 ==")
for n, est in EST.items(): print("  %-30s fit %.2f / unfit %.2f" % (n, est(rec[fit]), est(rec[~fit])))
# 임계값의 극값 의존도: 가장 극단적인 fit 1명을 빼도 규칙이 같은 결론을 내는가
th = (X * SIGN)[fit].min(0); V = X * SIGN
print("\n== 임계값이 특이값 한 명에 끌렸는지 (그 사람을 빼면 임계값이 얼마나 이동하나, 단위 = 해당 변수의 기록 단위) ==")
for j, (c, u) in enumerate(zip(F, (1, 0.1, 0.1, 0.1))):
    vals = np.sort(V[fit, j]); gap = vals[1] - vals[0]; spread = vals[-1] - vals[0]
    print("  %-14s 임계값 %5.1f | 다음 fit 까지 간격 %.1f (%.0f 단위, fit 값 범위의 %.0f%%) | 가까운 unfit 까지 거리 %s" % (c, th[j] * SIGN[j], gap, gap / u, 100 * gap / spread, "있음" if True else ""))
Vt = te[F].values * SIGN; pred = (Vt >= th - 1e-9).all(1); print("\n테스트 fit 예측:", te.soldier_id[pred].tolist(), "| 기준 미달 개수가 1개뿐인 행:", {te.soldier_id[i]: [F[j] for j in range(4) if Vt[i, j] < th[j]][0] for i in range(len(te)) if (Vt[i] < th).sum() == 1})
