import sys, os, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge, TheilSenRegressor, LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, DotProduct
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
try:
    from catboost import CatBoostRegressor, CatBoostClassifier; HAS_CB = True
except Exception: HAS_CB = False
R = 20; sp = lambda *s: make_pipeline(StandardScaler(), *s)

# =========================== P3 ===========================
t3 = pd.read_csv("problem03/data/train.csv"); F3 = ["leadership_score", "facility_score", "meal_score", "rest_time_score"]; X3 = t3[F3]; y3 = t3.overall_satisfaction.values
class Blend:
    def fit(self, X, y): self.a = LinearRegression().fit(X, y); self.m = LinearRegression().fit(np.asarray(X).mean(1).reshape(-1, 1), y); return self
    def predict(self, X): return (self.a.predict(X) + self.m.predict(np.asarray(X).mean(1).reshape(-1, 1))) / 2
def cvr(mk, X, y, reps=R):
    out = []
    for r in range(reps):
        oof = np.zeros(len(y))
        for a, b in KFold(5, shuffle=True, random_state=r).split(X): oof[b] = np.ravel(mk().fit(X.iloc[a], y[a]).predict(X.iloc[b]))
        out.append(np.sqrt(np.mean((y - oof) ** 2)))
    return np.array(out)
M3 = {"[메인] 선형+평균선형 평균": Blend, "선형 회귀": LinearRegression, "Ridge(1)": lambda: sp(Ridge(1.0)), "Lasso(0.01)": lambda: sp(Lasso(0.01)), "ElasticNet(0.02,0.5)": lambda: sp(ElasticNet(0.02, l1_ratio=0.5)),
      "BayesianRidge": BayesianRidge, "Huber": lambda: HuberRegressor(epsilon=1.5), "Theil-Sen": lambda: TheilSenRegressor(random_state=0), "PLS 2성분": lambda: PLSRegression(2), "PLS 1성분": lambda: PLSRegression(1),
      "SVR 선형 C=1": lambda: sp(SVR(kernel="linear", C=1.0, epsilon=0.1)), "SVR RBF C=10": lambda: sp(SVR(C=10, gamma=0.05, epsilon=0.1)), "KernelRidge RBF": lambda: sp(KernelRidge(alpha=0.1, kernel="rbf", gamma=0.02)),
      "가우시안 프로세스(선형+RBF)": lambda: sp(GaussianProcessRegressor(C(1.0) * DotProduct() + C(0.1) * RBF(3.0) + WhiteKernel(0.2), normalize_y=True, random_state=0)),
      "2차 다항 Ridge(1)": lambda: make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(1.0)), "kNN k=7": lambda: sp(KNeighborsRegressor(7)), "kNN k=15 거리가중": lambda: sp(KNeighborsRegressor(15, weights="distance")),
      "MLP(8) 은닉 1층": lambda: sp(MLPRegressor(hidden_layer_sizes=(8,), alpha=1.0, max_iter=3000, random_state=0)), "RandomForest": lambda: RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1), "ExtraTrees": lambda: ExtraTreesRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1),
      "GradientBoosting 깊이1": lambda: GradientBoostingRegressor(max_depth=1, n_estimators=400, learning_rate=0.05, subsample=0.8, random_state=0), "HistGradientBoosting": lambda: HistGradientBoostingRegressor(max_depth=2, min_samples_leaf=5, learning_rate=0.05, max_iter=300, random_state=0)}
if HAS_CB: M3["CatBoost 깊이2"] = lambda: CatBoostRegressor(depth=2, iterations=500, learning_rate=0.05, verbose=False, random_seed=0, allow_writing_files=False, thread_count=4)

# =========================== P5 ===========================
t5 = pd.read_csv("problem05/data/train.csv"); F5 = ["pushup", "run_time", "sleep_quality", "stress_level"]; X5 = t5[F5]; fit = (t5.status == "fit").values; rec = t5.recovery_days.values.astype(float); SIGN = np.array([1, -1, 1, -1])
class Box:
    def fit(self, X, y): self.th = (np.asarray(X, float) * SIGN)[np.asarray(y, bool)].min(0); return self
    def predict(self, X): return ((np.asarray(X, float) * SIGN) >= self.th).all(1)
cv5 = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=R, random_state=0).split(X5, fit))
def cvc(mk, reps=R):
    acc, rm = [], []
    for r in range(reps):
        ps = np.zeros(80, bool); oof = np.zeros(80)
        for a, b in cv5[r * 5:(r + 1) * 5]:
            pb = np.asarray(mk().fit(X5.iloc[a], fit[a]).predict(X5.iloc[b])).astype(bool); ps[b] = pb; oof[b] = np.where(pb, rec[a][fit[a]].mean(), rec[a][~fit[a]].mean())
        acc.append(np.mean(ps == fit)); rm.append(np.sqrt(np.mean((rec - oof) ** 2)))
    acc, rm = np.array(acc), np.array(rm); return acc, rm, 50 * acc + 50 / (1 + rm / 2.46475151)
M5 = {"[메인] 상자 규칙": Box, "로지스틱 C=1": lambda: sp(LogisticRegression(C=1, max_iter=2000)), "로지스틱 2차 다항 C=1": lambda: make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), LogisticRegression(C=1, max_iter=5000)),
      "LDA": lambda: sp(LinearDiscriminantAnalysis()), "QDA(reg 0.3)": lambda: sp(QuadraticDiscriminantAnalysis(reg_param=0.3)), "GaussianNB": GaussianNB, "kNN k=5": lambda: sp(KNeighborsClassifier(5)), "kNN k=3": lambda: sp(KNeighborsClassifier(3)),
      "SVM RBF C=10": lambda: sp(SVC(C=10, gamma="scale")), "SVM 선형 C=1": lambda: sp(SVC(kernel="linear", C=1)), "결정트리 깊이4": lambda: DecisionTreeClassifier(max_depth=4, random_state=0), "결정트리 무제한": lambda: DecisionTreeClassifier(random_state=0),
      "RandomForest": lambda: RandomForestClassifier(300, min_samples_leaf=1, random_state=0, n_jobs=-1), "ExtraTrees": lambda: ExtraTreesClassifier(300, random_state=0, n_jobs=-1), "GradientBoosting 깊이2": lambda: GradientBoostingClassifier(max_depth=2, n_estimators=200, learning_rate=0.05, random_state=0),
      "MLP(8)": lambda: sp(MLPClassifier(hidden_layer_sizes=(8,), alpha=0.1, max_iter=5000, random_state=0))}
if HAS_CB: M5["CatBoost 깊이3"] = lambda: CatBoostClassifier(depth=3, iterations=300, learning_rate=0.05, verbose=False, random_seed=0, allow_writing_files=False, thread_count=4)

if __name__ == "__main__":
    print("== P3: 모델 계열 비교 (5-Fold x 20 RMSE, 점수 = 100/(1+RMSE/1.0589)) ==")
    res = {}
    for n, mk in M3.items():
        reps = 5 if ("MLP" in n or "가우시안" in n or "Theil" in n or "CatBoost" in n) else R
        res[n] = cvr(mk, X3, y3, reps)
    base = res["[메인] 선형+평균선형 평균"]
    for n, v in sorted(res.items(), key=lambda kv: kv[1].mean()):
        d = v - base[:len(v)]; print(f"  {n:28s} RMSE {v.mean():.4f}  점수 {100 / (1 + v.mean() / 1.05889019):.2f}  메인 대비 {d.mean():+.4f} ({int((d < 0).sum())}/{len(v)} 반복에서 더 좋음)", flush=True)
    print("\n== P5: status 모델 계열 비교 (Stratified 5-Fold x 20; 회복일은 예측 status 의 그룹 평균) ==")
    res5 = {}
    for n, mk in M5.items():
        reps = 5 if ("MLP" in n or "CatBoost" in n or "GradientBoosting" in n) else R
        res5[n] = cvc(mk, reps)
    b5 = res5["[메인] 상자 규칙"][2]
    for n, (acc, rm, sc) in sorted(res5.items(), key=lambda kv: -kv[1][2].mean()):
        d = sc - b5[:len(sc)]; print(f"  {n:24s} 정확도 {acc.mean():.4f}  회복 RMSE {rm.mean():.4f}  점수 {sc.mean():.2f}  메인 대비 {d.mean():+.2f} ({int((d > 0).sum())}/{len(sc)} 반복에서 더 좋음)", flush=True)
