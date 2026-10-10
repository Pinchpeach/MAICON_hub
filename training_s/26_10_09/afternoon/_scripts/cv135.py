import pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

R = 20
def load(p): return pd.read_csv(f"problem{p}/data/train.csv")


def cv_reg(X, y, mk, post=None, name=""):
    """RepeatedKFold(5 x R) RMSE. 반복마다 전체 OOF RMSE 를 구한 뒤 평균·표준편차를 보고, 반복별 RMSE 벡터를 반환."""
    rm = []
    for r in range(R):
        oof = np.zeros(len(y));
        from sklearn.model_selection import KFold
        for a, b in KFold(5, shuffle=True, random_state=r).split(X):
            m = mk().fit(X.iloc[a], y.iloc[a]); p = m.predict(X.iloc[b]); oof[b] = post(p) if post else p
        rm.append(np.sqrt(np.mean((y.values - oof) ** 2)))
    rm = np.array(rm); print(f"  {name:44s} RMSE {rm.mean():.4f} (반복 sd {rm.std():.4f})", flush=True); return rm


def paired(a, b, la, lb):
    d = a - b; print(f"    [{la}] - [{lb}] RMSE 차 평균 {d.mean():+.4f} (반복 sd {d.std():.4f}, 반복 {int((d < 0).sum())}/{len(d)} 에서 앞이 더 좋음)")


# ---------------- P1 ----------------
t = load("01"); y = t.fatigue_score; F = ["avg_sleep_hours", "night_shift_days", "caffeine_intake"]
print("== P1 (피로도, RMSE 낮을수록; 점수=100/(1+RMSE/9.6565)) ==")
clip = lambda p: np.clip(p, 30.0, None)
base = cv_reg(t[F], y, LinearRegression, None, "선형")
lc = cv_reg(t[F], y, LinearRegression, clip, "선형 + 하한 30 클리핑")
paired(lc, base, "클리핑", "선형")
def feat(df, thr_c=None, cap_n=None):
    X = df[F].copy()
    if thr_c: X["caffeine_intake"] = np.minimum(X["caffeine_intake"], thr_c)
    if cap_n: X["night_shift_days"] = np.minimum(X["night_shift_days"], cap_n)
    return X
for thr in (100, 125, 150):
    cv_reg(feat(t, thr_c=thr), y, LinearRegression, clip, f"선형+클립, 카페인 min(c,{thr})")
for cap in (3, 4):
    cv_reg(feat(t, cap_n=cap), y, LinearRegression, clip, f"선형+클립, 야간 min(n,{cap})")
cv_reg(feat(t, thr_c=125, cap_n=3), y, LinearRegression, clip, "선형+클립, 카페인125·야간3 동시")
cv_reg(t[F], y, lambda: make_pipeline(StandardScaler(), Ridge(alpha=3.0)), clip, "Ridge(3)+클립")
cv_reg(t[F], y, lambda: make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=5.0)), clip, "2차 다항 Ridge(5)+클립")
cv_reg(t[F], y, lambda: RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1), clip, "RF(참고)")
cv_reg(t[F], y, lambda: HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_depth=3, random_state=0), clip, "HGB(참고)")

# ---------------- P3 ----------------
t = load("03"); y = t.overall_satisfaction; F = ["leadership_score", "facility_score", "meal_score", "rest_time_score"]
print("\n== P3 (만족도, 점수=100/(1+RMSE/1.0589)) ==")
class MeanModel:
    def fit(self, X, y): self.b = (y - X.mean(axis=1)).mean(); return self
    def predict(self, X): return X.mean(axis=1).values + self.b
b3 = cv_reg(t[F], y, LinearRegression, None, "선형")
cv_reg(t[F], y, MeanModel, None, "4항목 평균 + 상수")
cv_reg(t[F], y, lambda: make_pipeline(StandardScaler(), Ridge(alpha=3.0)), None, "Ridge(3)")
cv_reg(t[F], y, lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)), None, "Ridge(10)")
cv_reg(t[F], y, lambda: make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=10.0)), None, "2차 다항 Ridge(10)")
cv_reg(t[F], y, lambda: RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1), None, "RF(참고)")
cv_reg(t[F], y, lambda: HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_depth=3, random_state=0), None, "HGB(참고)")
cv_reg(t[F], y, LinearRegression, lambda p: np.clip(p, 1, 10), "선형 + [1,10] 클리핑")

# ---------------- P5 ----------------
t = load("05"); F = ["pushup", "run_time", "sleep_quality", "stress_level"]; X = t[F]; fit = (t.status == "fit").astype(int).values; rec = t.recovery_days.values.astype(float)
print("\n== P5 (점수 = 50*정확도 + 50/(1+RMSE/2.4648)) ==")
cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=R, random_state=0).split(X, fit))
def clf_cv(mk, name):
    accs = []
    P = np.zeros((R, len(t)))
    for i, (a, b) in enumerate(cv):
        r = i // 5; m = mk().fit(X.iloc[a], fit[a]); P[r, b] = m.predict_proba(X.iloc[b])[:, 1]
    acc = np.array([np.mean((P[r] >= 0.5) == fit) for r in range(R)]); print(f"  {name:44s} 정확도 {acc.mean():.4f} (반복 sd {acc.std():.4f})", flush=True); return P, acc
sc = lambda C: (lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)))
Ps = {}
print(" -- status --")
print(f"  다수 클래스 기준 정확도 {max(fit.mean(), 1 - fit.mean()):.4f}")
for C in (0.1, 0.3, 1, 3, 10): Ps[f"LR C={C}"] = clf_cv(sc(C), f"로지스틱 C={C}")
Ps["LR bal"] = clf_cv(lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, class_weight="balanced")), "로지스틱 C=1 balanced")
Ps["LDA"] = clf_cv(lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()), "LDA")
Ps["RF"] = clf_cv(lambda: RandomForestClassifier(300, min_samples_leaf=2, random_state=0, n_jobs=-1), "RF(참고)")
print(" -- recovery (정답 status 를 안다고 가정한 상한과 실제 예측 status 사용 두 가지) --")
def rec_eval(P_status, kind, name):
    rm = []
    for r in range(R):
        oof = np.zeros(len(t)); idx = [(a, b) for i, (a, b) in enumerate(cv) if i // 5 == r]
        for a, b in idx:
            p = P_status[r, b]
            if kind == "direct_lr": m = LinearRegression().fit(X.iloc[a], rec[a]); oof[b] = m.predict(X.iloc[b])
            elif kind == "direct_rf": m = RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(X.iloc[a], rec[a]); oof[b] = m.predict(X.iloc[b])
            elif kind == "const": oof[b] = rec[a].mean()
            elif kind in ("mix_const", "mix_lin"):
                fa = fit[a] == 1; mf = rec[a][fa].mean()
                if kind == "mix_const": mu = rec[a][~fa].mean(); oof[b] = p * mf + (1 - p) * mu
                else:
                    m = LinearRegression().fit(X.iloc[a][~fa], rec[a][~fa]); mu = m.predict(X.iloc[b]); oof[b] = p * mf + (1 - p) * mu
            elif kind == "oracle_const": fa = fit[a] == 1; oof[b] = np.where(fit[b] == 1, rec[a][fa].mean(), rec[a][~fa].mean())
        rm.append(np.sqrt(np.mean((rec - oof) ** 2)))
    rm = np.array(rm); print(f"  {name:44s} RMSE {rm.mean():.4f} (반복 sd {rm.std():.4f})", flush=True); return rm
Pst = Ps["LR C=1"][0]
rec_eval(Pst, "const", "상수(평균)"); rec_eval(Pst, "direct_lr", "특징 선형 회귀"); rec_eval(Pst, "direct_rf", "특징 RF 회귀")
rec_eval(Pst, "mix_const", "2단계: p*fit평균 + (1-p)*unfit평균"); rec_eval(Pst, "mix_lin", "2단계: unfit 은 특징 선형 회귀")
rec_eval(Pst, "oracle_const", "[상한] 정답 status 알 때 그룹 평균")
