import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
t = pd.read_csv("problem05/data/train.csv"); F = ["pushup", "run_time", "sleep_quality", "stress_level"]; X = t[F].values.astype(float); fit = (t.status == "fit").values; rec = t.recovery_days.values.astype(float); SIGN = np.array([1, -1, 1, -1]); SC = 2.46475151; R = 20
cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=R, random_state=0).split(X, fit))
mode = lambda v: float(pd.Series(v).mode().iloc[0])
EST = {"평균 (메인)": (np.mean, np.mean), "평균을 반올림": (lambda v: np.round(np.mean(v)), lambda v: np.round(np.mean(v))), "중앙값": (np.median, np.median), "최빈값": (mode, mode),
       "fit 은 평균, unfit 만 반올림": (np.mean, lambda v: np.round(np.mean(v))), "fit 은 평균, unfit 은 중앙값": (np.mean, np.median), "fit 은 내림(5), unfit 평균": (lambda v: np.floor(np.mean(v)), np.mean), "fit 은 올림(6), unfit 평균": (lambda v: np.ceil(np.mean(v)), np.mean),
       "fit 은 중앙값, unfit 평균": (np.median, np.mean)}
def run(ff, fu, oracle=False):
    acc, rm = [], []
    for r in range(R):
        ps = np.zeros(80, bool); oof = np.zeros(80)
        for a, b in cv[r * 5:(r + 1) * 5]:
            th = (X[a] * SIGN)[fit[a]].min(0); pb = fit[b] if oracle else ((X[b] * SIGN) >= th).all(1); ps[b] = pb
            oof[b] = np.where(pb, ff(rec[a][fit[a]]), fu(rec[a][~fit[a]]))
        acc.append(np.mean(ps == fit)); rm.append(np.sqrt(np.mean((rec - oof) ** 2)))
    acc, rm = np.array(acc), np.array(rm); return rm, 50 * acc + 50 / (1 + rm / SC)
print("== 회복일 예측값을 정하는 방법 (Stratified 5-Fold x 20, status 는 상자 규칙) ==")
base_rm, base_sc = run(np.mean, np.mean)
for n, (ff, fu) in EST.items():
    rm, sc = run(ff, fu); d = sc - base_sc; print("  %-30s 회복 RMSE %.4f  점수 %.3f  메인 대비 %+.3f (%2d/20 반복에서 더 좋음)" % (n, rm.mean(), sc.mean(), d.mean(), int((d > 0).sum())))
print("\n== 같은 비교, status 를 모두 맞혔다고 가정 (회복일 부분만 분리) ==")
o_rm, o_sc = run(np.mean, np.mean, oracle=True)
for n in ("평균 (메인)", "평균을 반올림", "중앙값", "fit 은 평균, unfit 만 반올림", "fit 은 내림(5), unfit 평균", "fit 은 올림(6), unfit 평균"):
    rm, sc = run(*EST[n], oracle=True); print("  %-30s 회복 RMSE %.4f  메인 대비 %+.4f" % (n, rm.mean(), (rm - o_rm).mean()))
# 전체 학습 기준의 값과, 예측 상수 c 를 썼을 때 학습 데이터에서의 RMSE
mf, mu = rec[fit].mean(), rec[~fit].mean()
print("\n전체 학습: fit 평균 %.2f / 중앙값 %.1f / 최빈값 %.0f (10명) | unfit 평균 %.3f / 중앙값 %.1f / 최빈값 %.0f (70명)" % (mf, np.median(rec[fit]), mode(rec[fit]), mu, np.median(rec[~fit]), mode(rec[~fit])))
print("unfit 에 상수 c 를 썼을 때 학습 RMSE:", {c: round(float(np.sqrt(np.mean((rec[~fit] - c) ** 2))), 4) for c in (9, 10, round(mu, 2), 11)})
print("fit 에 상수 c 를 썼을 때 학습 RMSE:", {c: round(float(np.sqrt(np.mean((rec[fit] - c) ** 2))), 4) for c in (5, 5.5, 6)})
# 참 평균을 모르는 상태에서 10.07 대신 10 을 쓰는 것이 유리할 확률 (부트스트랩)
rng = np.random.default_rng(0); B = 20000; bm = np.array([rng.choice(rec[~fit], 70).mean() for _ in range(B)])
print("unfit 참 평균의 부트스트랩 분포: 평균 %.3f, 표준오차 %.3f, 95%% 구간 %.2f ~ %.2f" % (bm.mean(), bm.std(), *np.percentile(bm, [2.5, 97.5])))
bf = np.array([rng.choice(rec[fit], 10).mean() for _ in range(B)]); print("fit 참 평균의 부트스트랩 분포: 평균 %.3f, 표준오차 %.3f, 95%% 구간 %.2f ~ %.2f" % (bf.mean(), bf.std(), *np.percentile(bf, [2.5, 97.5])))
print("테스트 21행에서의 영향: unfit 예측 10.07 -> 10 이면 RMSE 변화 최대 약 %.4f, fit 1명 5.5 -> 5 또는 6 이면 그 행 오차만 0.5 이동" % abs(np.sqrt(mu * 0 + 2.05 ** 2 + 0.07 ** 2) - 2.05))
