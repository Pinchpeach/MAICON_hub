import sys, os, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor as GBR, HistGradientBoostingRegressor as HGB
from sklearn.linear_model import LinearRegression
from cv135b import Tobit, cv_reg, paired
t = pd.read_csv("problem01/data/train.csv"); y = t.fatigue_score; F = ["avg_sleep_hours", "night_shift_days", "caffeine_intake"]; X = t[F]
clip = lambda p: np.clip(p, 30, None)
print("== P1 부스팅 트리 실험 (5-Fold x 20회 반복, 모두 예측을 하한 30으로 자름) ==")
base = cv_reg(X, y, Tobit, None, "Tobit (메인)")
res = {}
# 1) sklearn GradientBoosting: 깊이·학습률·나무 수
for d in (1, 2, 3):
    for lr, n in ((0.05, 200), (0.05, 500), (0.1, 300), (0.02, 1000)):
        res[f"GBR d{d} lr{lr} n{n}"] = cv_reg(X, y, lambda d=d, lr=lr, n=n: GBR(max_depth=d, learning_rate=lr, n_estimators=n, subsample=0.8, random_state=0), clip, f"GBR 깊이{d} lr{lr} 나무{n}")
# 2) HistGradientBoosting (작은 데이터라 min_samples_leaf 를 낮춤)
for d, leaf, lr, it in ((2, 5, 0.05, 300), (3, 5, 0.05, 300), (2, 3, 0.1, 200), (None, 10, 0.05, 200)):
    res[f"HGB d{d} leaf{leaf}"] = cv_reg(X, y, lambda d=d, leaf=leaf, lr=lr, it=it: HGB(max_depth=d, min_samples_leaf=leaf, learning_rate=lr, max_iter=it, l2_regularization=1.0, random_state=0), clip, f"HGB 깊이{d} 잎최소{leaf} lr{lr} 반복{it}")
# 3) 단조 제약 (수면↓, 야간↑, 카페인↑)
res["HGB mono"] = cv_reg(X, y, lambda: HGB(max_depth=2, min_samples_leaf=5, learning_rate=0.05, max_iter=300, monotonic_cst=[-1, 1, 1], random_state=0), clip, "HGB 깊이2 + 단조 제약")
# 4) CatBoost (로컬에만 설치, 서버에는 없음)
try:
    from catboost import CatBoostRegressor as CB
    for d in (2, 4):
        res[f"CB d{d}"] = cv_reg(X, y, lambda d=d: CB(depth=d, iterations=500, learning_rate=0.05, verbose=False, random_seed=0, allow_writing_files=False, thread_count=4), clip, f"CatBoost 깊이{d} 500회")
except Exception as e: print("  CatBoost 사용 불가:", e)
# 5) 혼합: 선형/Tobit 이 큰 구조를 잡고 부스팅이 잔차를 학습
class ResidBoost:
    def __init__(self, base="tobit"): self.base = base
    def fit(self, X, y):
        self.m = Tobit().fit(X, y) if self.base == "tobit" else LinearRegression().fit(X, y)
        r = np.asarray(y, float) - self.m.predict(X); self.g = GBR(max_depth=1, learning_rate=0.03, n_estimators=100, subsample=0.8, random_state=0).fit(X, r); return self
    def predict(self, X): return np.clip(self.m.predict(X) + self.g.predict(X), 30, None)
res["Tobit+GBR resid"] = cv_reg(X, y, lambda: ResidBoost("tobit"), None, "Tobit + 잔차 부스팅(깊이1, 100그루)")
res["Lin+GBR resid"] = cv_reg(X, y, lambda: ResidBoost("lin"), None, "선형 + 잔차 부스팅(깊이1, 100그루)")
# 6) Tobit 과 최선 부스팅의 평균
best = min((k for k in res if "resid" not in k), key=lambda k: res[k].mean()); print("\n부스팅 단독 최선:", best, "RMSE %.4f" % res[best].mean())
paired(res[best], base, best, "Tobit"); paired(res["Tobit+GBR resid"], base, "Tobit+잔차부스팅", "Tobit")
