import sys, os, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy.stats import norm, shapiro, kstest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_val_predict
from cv135b import Tobit
pd.set_option("display.width", 250)
L = lambda p, f="train": pd.read_csv(f"problem{p}/data/{f}.csv")
t1, t3, t5 = L("01"), L("03"), L("05"); idn = lambda d: d.soldier_id.str[1:].astype(int)
F1 = ["avg_sleep_hours", "night_shift_days", "caffeine_intake"]; F3 = ["leadership_score", "facility_score", "meal_score", "rest_time_score"]; F5 = ["pushup", "run_time", "sleep_quality", "stress_level"]
print("== 0. ID 구조 ==")
for n, d, p in (("P1", t1, "01"), ("P3", t3, "03"), ("P5", t5, "05")):
    te = L(p, "test"); print(n, "train ID 범위", idn(d).min(), idn(d).max(), "| test", idn(te).min(), idn(te).max(), "| 합쳐서 고유", len(set(idn(d)) | set(idn(te))), "| train ID 정렬됨", idn(d).is_monotonic_increasing)
# ---------- P1 ----------
print("\n== P1 잡음 하한 ==")
m = Tobit().fit(t1[F1], t1.fatigue_score); Z = np.c_[np.ones(80), (t1[F1].values - m.m) / m.s]; mu = Z @ m.b; sg = m.sg
rng = np.random.default_rng(0); sims = []
for _ in range(4000):
    y = np.maximum(30, mu + rng.normal(0, sg, 80)); z = (mu - 30) / sg; pred = 30 + (mu - 30) * norm.cdf(z) + sg * norm.pdf(z); sims.append(np.sqrt(np.mean((y - pred) ** 2)))
print("  참 모수를 안다고 가정한 예측의 RMSE(모의 4000회): 평균 %.3f, 5~95%% %.3f~%.3f  (현재 CV 2.843)" % (np.mean(sims), *np.percentile(sims, [5, 95])))
unc = t1.fatigue_score.values > 30; res = t1.fatigue_score.values[unc] - mu[unc]
print("  비검열 잔차 %d개: sd %.2f, 왜도 %.2f, 첨도 %.2f, Shapiro p=%.3f | 잔차-ID 상관 %.3f" % (unc.sum(), res.std(), pd.Series(res).skew(), pd.Series(res).kurt(), shapiro(res)[1], np.corrcoef(res, idn(t1)[unc])[0, 1]))
print("  잔차 소수 첫째 자리 분포:", pd.Series(np.round(res, 1) * 10 % 10).round().value_counts().sort_index().to_dict())
X1 = t1[F1].assign(idn=idn(t1), mu=mu); rf = RandomForestRegressor(500, min_samples_leaf=5, random_state=0, n_jobs=-1)
r_oof = cross_val_predict(rf, X1[unc], res, cv=KFold(5, shuffle=True, random_state=0)); print("  잔차를 RF 로 예측(비검열 행, 5-fold): R2 %.3f (0 이하면 남은 신호 없음)" % (1 - np.mean((res - r_oof) ** 2) / res.var()))
# ---------- P3 ----------
print("\n== P3 잡음 하한 ==")
lm = LinearRegression().fit(t3[F3], t3.overall_satisfaction); r3 = t3.overall_satisfaction.values - lm.predict(t3[F3]); s_unb = np.sqrt((r3 ** 2).sum() / (80 - 5))
print("  불편 잡음 추정 sigma %.4f -> 참 모수 예측의 RMSE 약 %.4f (현재 CV 0.4585) | Shapiro p=%.3f | 잔차-ID 상관 %.3f" % (s_unb, s_unb, shapiro(r3)[1], np.corrcoef(r3, idn(t3))[0, 1]))
D = t3[F3]; extra = pd.DataFrame({"min": D.min(1), "max": D.max(1), "std": D.std(1), "range": D.max(1) - D.min(1), "idn": idn(t3), "n_low(<5)": (D < 5).sum(1), "n_high(>=8)": (D >= 8).sum(1), "prod_LM": D.leadership_score * D.meal_score})
print("  잔차와 파생 특징 상관:", {c: round(float(np.corrcoef(r3, extra[c])[0, 1]), 3) for c in extra})
r_oof = cross_val_predict(RandomForestRegressor(500, min_samples_leaf=5, random_state=0, n_jobs=-1), pd.concat([D, extra], axis=1), r3, cv=KFold(5, shuffle=True, random_state=0)); print("  잔차를 RF 로 예측: R2 %.3f" % (1 - np.mean((r3 - r_oof) ** 2) / r3.var()))
print("  타깃 x10 이 정수인가:", np.allclose(t3.overall_satisfaction * 10, np.round(t3.overall_satisfaction * 10)), "| 입력 x10 정수:", np.allclose(D * 10, np.round(D * 10)))
# ---------- P5 ----------
print("\n== P5 회복일 구조 ==")
fit = (t5.status == "fit").values; u = t5[~fit]; ru = u.recovery_days.values.astype(float)
print("  unfit 70명 회복일 sd %.3f, fit 10명 sd %.3f -> status 를 다 맞혀도 RMSE 하한 약 %.3f" % (ru.std(), t5[fit].recovery_days.std(ddof=0), np.sqrt((70 * ru.var() + 10 * t5[fit].recovery_days.var(ddof=0)) / 80)))
Xu = u[F5].assign(idn=idn(u)); SIGN = np.array([1, -1, 1, -1]); th = np.array([38, -9.6, 5.2, -5.8]); V = u[F5].values * SIGN
ex = pd.DataFrame({"nfail": (V < th).sum(1), "gap_sum": np.clip(th - V, 0, None).sum(1), "gap_push": np.clip(38 - u.pushup, 0, None), "gap_run": np.clip(u.run_time - 9.6, 0, None), "gap_sleep": np.clip(5.2 - u.sleep_quality, 0, None), "gap_stress": np.clip(u.stress_level - 5.8, 0, None), "idn": idn(u).values}, index=u.index)
print("  unfit 회복일과 상관:", {c: round(float(np.corrcoef(ru, pd.concat([u[F5], ex], axis=1)[c])[0, 1]), 3) for c in F5 + list(ex.columns)})
for nm, mdl in (("RF", RandomForestRegressor(500, min_samples_leaf=5, random_state=0, n_jobs=-1)), ("선형", LinearRegression())):
    o = cross_val_predict(mdl, pd.concat([u[F5], ex], axis=1), ru, cv=KFold(5, shuffle=True, random_state=0)); print("  unfit 회복일을 %s 로 예측: R2 %.3f" % (nm, 1 - np.mean((ru - o) ** 2) / ru.var()))
print("  unfit 회복일 분포:", pd.Series(ru).value_counts().sort_index().to_dict())
# ---------- 문제 간 ID 연결 ----------
print("\n== 문제 간 같은 soldier_id 연결 (진단용, 사용 여부는 별도 판단) ==")
a = pd.concat([t1, L("01", "test")]).set_index("soldier_id"); b = pd.concat([t3, L("03", "test")]).set_index("soldier_id"); c = pd.concat([t5, L("05", "test")]).set_index("soldier_id")
j = a.join(b, how="inner", rsuffix="_3").join(c, how="inner", rsuffix="_5"); print("  세 문제 공통 ID 수:", len(j))
num = j.select_dtypes("number"); cm = num.corr()
pairs = [(x, y, cm.loc[x, y]) for x in a.select_dtypes("number").columns for y in list(b.select_dtypes("number").columns) + list(c.select_dtypes("number").columns)] + [(x, y, cm.loc[x, y]) for x in b.select_dtypes("number").columns for y in c.select_dtypes("number").columns]
print("  문제 간 변수 상관 상위 6:", [(x, y, round(float(v), 3)) for x, y, v in sorted(pairs, key=lambda q: -abs(q[2]))[:6]])
