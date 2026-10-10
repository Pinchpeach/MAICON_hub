import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
t = pd.read_csv("problem05/data/train.csv"); te = pd.read_csv("problem05/data/test.csv"); F = ["pushup", "run_time", "sleep_quality", "stress_level"]
fit = (t.status == "fit").values; u = t[~fit].reset_index(drop=True); ru = u.recovery_days.values.astype(float); SC = 2.46475151
# ---------- A. unfit 안에서 회복일수를 설명하는 모델이 있는가 (5-Fold x 20) ----------
print("== A. unfit 70명 회복일수: 그룹 평균 대비 (RMSE, 낮을수록 좋음) ==")
def cv(mk, X):
    out = []
    for r in range(20):
        oof = np.zeros(len(ru))
        for a, b in KFold(5, shuffle=True, random_state=r).split(X): oof[b] = mk().fit(X.iloc[a], ru[a]).predict(X.iloc[b])
        out.append(np.sqrt(np.mean((ru - oof) ** 2)))
    return np.array(out)
class Const:
    def fit(self, X, y): self.m = np.mean(y); return self
    def predict(self, X): return np.full(len(X), self.m)
class Med(Const):
    def fit(self, X, y): self.m = np.median(y); return self
Xu = u[F]; G = Xu.assign(gp=np.clip(38 - u.pushup, 0, None), gr=np.clip(u.run_time - 9.6, 0, None), gs=np.clip(5.2 - u.sleep_quality, 0, None), gt=np.clip(u.stress_level - 5.8, 0, None))
G["nfail"] = (G[["gp", "gr", "gs", "gt"]] > 0).sum(1)
base = cv(Const, Xu); print("  %-40s %.4f" % ("그룹 평균 (현재)", base.mean()))
cands = {"그룹 중앙값": (Med, Xu), "Ridge(10) 특징 4개": (lambda: make_pipeline(StandardScaler(), Ridge(10.0)), Xu), "Ridge(30) 특징+미달폭": (lambda: make_pipeline(StandardScaler(), Ridge(30.0)), G),
         "kNN k=10": (lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(10)), Xu), "kNN k=25": (lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(25)), Xu),
         "RF 잎최소 10": (lambda: RandomForestRegressor(300, min_samples_leaf=10, random_state=0, n_jobs=-1), G), "부스팅 깊이1 50그루 lr0.03": (lambda: GradientBoostingRegressor(max_depth=1, n_estimators=50, learning_rate=0.03, random_state=0), G),
         "스트레스 한 변수 선형": (LinearRegression, u[["stress_level"]]), "미달 개수 한 변수 선형": (LinearRegression, G[["nfail"]])}
for n, (mk, X) in cands.items():
    r = cv(mk, X); d = r - base; print("  %-40s %.4f  (평균 대비 %+.4f, %2d/20 반복에서 더 좋음)" % (n, r.mean(), d.mean(), int((d < 0).sum())))
# ---------- B. S035 처리별 기대 점수 ----------
print("\n== B. 테스트 21명 기대 점수 모의 (S035 가 fit 일 확률 p 에 따라) ==")
mf, mu = t.recovery_days[fit].mean(), ru.mean(); sf, su = t.recovery_days[fit].std(ddof=0), ru.std()
rng = np.random.default_rng(0); N = 20000
def expected(p_true, status_choice, rec_pred):
    """S066 은 fit, 나머지 19명은 unfit 으로 맞혔다고 가정. S035 만 불확실"""
    is_fit = rng.random(N) < p_true
    r35 = np.where(is_fit, rng.normal(mf, sf, N), rng.normal(mu, su, N)); e35 = (r35 - rec_pred) ** 2
    e66 = (rng.normal(mf, sf, N) - mf) ** 2; e19 = ((rng.normal(mu, su, (N, 19)) - mu) ** 2).sum(1)
    rmse = np.sqrt((e35 + e66 + e19) / 21); acc = (20 + ((status_choice == "fit") == is_fit)) / 21
    return (50 * acc + 50 / (1 + rmse / SC)).mean()
print("  p     | unfit·10.07일(현재) | unfit·확률가중 | fit·5.5일 | fit·확률가중")
for p in (0.2, 0.43, 0.5, 0.6, 0.8):
    h = p * mf + (1 - p) * mu; print("  %.2f  |      %.2f         |     %.2f      |   %.2f   |    %.2f" % (p, expected(p, "unfit", mu), expected(p, "unfit", h), expected(p, "fit", mf), expected(p, "fit", h)))
print("  (가중 회복일: p=0.43 -> %.2f일, p=0.5 -> %.2f일. 참고: 모두 맞히고 잡음만 남을 때 약 %.2f)" % (0.43 * mf + 0.57 * mu, 0.5 * mf + 0.5 * mu, expected(0.0, "unfit", mu)))
# ---------- C. fit 그룹 평균의 불확실성 ----------
print("\n== C. 그룹 평균의 표본 오차 ==")
print("  fit 평균 %.2f (10명, 표준오차 %.2f) | unfit 평균 %.2f (70명, 표준오차 %.2f)" % (mf, sf / np.sqrt(10), mu, su / np.sqrt(70)))
print("  fit 회복일 값:", sorted(t.recovery_days[fit].tolist()), "| 전체 최솟값", int(t.recovery_days.min()))
