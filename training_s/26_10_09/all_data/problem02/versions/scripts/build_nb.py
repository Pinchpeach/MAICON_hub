import nbformat, os, shutil
from nbclient import NotebookClient

os.makedirs("versions", exist_ok=True)
if not os.path.exists("versions/notebook_orig.ipynb"): shutil.copy("work.ipynb", "versions/notebook_orig.ipynb")
nb = nbformat.read("versions/notebook_orig.ipynb", as_version=4)

md = nbformat.v4.new_markdown_cell("""## 풀이: uid 내 과거 정보 특징 + 계층 분류(A) + RF·HGB 확률평균 + 소수 클래스 가중
- 특징 59개: 각속도·가속도 크기, 부호 있는 z축 가속도 평균, 배터리 전압·전류 기울기와 비행 시작 대비 변화, 창 5/10/50/100/200행 평균·표준편차·최대·최소, 지수이동평균. **모두 같은 uid 안에서 현재와 과거 행만 사용**(미래 값 미사용). 원값 센서(전압·전류·바람)는 비행 간 분포 이동 때문에 넣으면 Macro F1이 떨어져 제외(0.69 → 0.64).
- 구조 A: 1단계 IDLE/HMSL/TURN/(ASCEND+DESCEND 묶음) → 2단계 묶음 안에서 ASCEND/DESCEND. DESCEND 전용 먼저 거르는 구조(B)는 거짓 양성이 늘어 더 낮았음.
- RF와 HGB의 확률을 평균하고, ASCEND×3, DESCEND×5 로 확률을 가중해 argmax. **가중치는 OOF 점수로 골라 낙관적**이며 DESCEND 구간이 31개뿐이라 일반화는 장담할 수 없음.
- 검증: 비행(uid) 단위 Leave-One-Flight-Out 5폴드 OOF. 과거 지수평활은 오히려 해로워 제외.
- 비용: 이 셀 전체가 개발 PC(16스레드)에서 약 2분. 서버 2코어에서는 4~8배(추정, 미측정)로 본다.""")

code = nbformat.v4.new_code_cell('''import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

CL = ["IDLE_HOVER", "ASCEND", "DESCEND", "TURN", "HMSL"]
W = np.array([1, 3, 5, 1, 1.0])                      # CL 순서 가중 (ASCEND x3, DESCEND x5)
tr = train.sort_values(["uid", "seq"]).reset_index(drop=True)
te = test.sort_values(["uid", "seq"]).reset_index(drop=True)

def make_features(df):
    """uid 안에서 현재+과거 행만 쓰는 특징 (미래 값 미사용)"""
    uid = df.uid.astype(str); g = df.groupby(uid, sort=False); F = pd.DataFrame(index=df.index)
    ang = np.sqrt(df.angular_x ** 2 + df.angular_y ** 2 + df.angular_z ** 2)
    acc = np.sqrt(df.linear_acceleration_x ** 2 + df.linear_acceleration_y ** 2 + df.linear_acceleration_z ** 2)
    az = df.linear_acceleration_z
    F["abs_ang_x"], F["abs_ang_y"], F["abs_ang_z"] = df.angular_x.abs(), df.angular_y.abs(), df.angular_z.abs()
    F["ang_norm"], F["acc_norm"] = ang, acc
    for c in ("linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"): F["abs_" + c] = df[c].abs()
    def roll(s, n, fn): return getattr(s.groupby(uid, sort=False).rolling(n, min_periods=2), fn)().reset_index(level=0, drop=True).sort_index()
    def ewm(s, span): return s.groupby(uid, sort=False).transform(lambda x: x.ewm(span=span, adjust=False).mean())
    for n in (5, 10, 50, 100, 200):
        F[f"ang_norm_m{n}"] = roll(ang, n, "mean"); F[f"acc_norm_m{n}"] = roll(acc, n, "mean"); F[f"acc_z_m{n}"] = roll(az, n, "mean")   # acc_z 는 부호 포함
    for n in (5, 10, 50): F[f"abs_ang_z_m{n}"] = roll(df.angular_z.abs(), n, "mean")
    for n in (10, 50):
        F[f"ang_norm_s{n}"] = roll(ang, n, "std"); F[f"acc_norm_s{n}"] = roll(acc, n, "std"); F[f"acc_z_s{n}"] = roll(az, n, "std")
        F[f"acc_norm_max{n}"] = roll(acc, n, "max"); F[f"acc_z_max{n}"] = roll(az, n, "max"); F[f"acc_z_min{n}"] = roll(az, n, "min")
    F["ang_z_s10"] = roll(df.angular_z, 10, "std"); F["acc_x_s10"] = roll(df.linear_acceleration_x, 10, "std"); F["acc_y_s10"] = roll(df.linear_acceleration_y, 10, "std")
    for sp in (5, 20, 100): F[f"acc_norm_e{sp}"] = ewm(acc, sp); F[f"ang_norm_e{sp}"] = ewm(ang, sp); F[f"acc_z_e{sp}"] = ewm(az, sp)
    for n in (10, 50, 100): F[f"cur_slope{n}"] = (df.battery_current - g.battery_current.shift(n)) / n
    for n in (50, 100): F[f"volt_slope{n}"] = (df.battery_voltage - g.battery_voltage.shift(n)) / n
    F["volt_rel"] = df.battery_voltage - g.battery_voltage.transform("first")      # 비행 시작 대비
    F["cur_m50"] = roll(df.battery_current, 50, "mean")
    F["acc_trend"] = F["acc_norm_m10"] - F["acc_norm_m50"]; F["ang_trend"] = F["ang_norm_m10"] - F["ang_norm_m50"]
    return F.fillna(0.0)

def rf(**k): return RandomForestClassifier(300, min_samples_leaf=3, n_jobs=-1, random_state=0, **k)
def hgb(**k): return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, random_state=0, **k)

def probs_A(mk, X, y, Xt):
    """계층 A: P(IDLE,HMSL,TURN,AD) x P(ASC|AD), P(DESC|AD)"""
    y = np.asarray(y); y1 = np.where(np.isin(y, ["ASCEND", "DESCEND"]), "AD", y)
    m1 = mk().fit(X, y1); P1 = pd.DataFrame(m1.predict_proba(Xt), columns=m1.classes_)
    m = np.isin(y, ["ASCEND", "DESCEND"]); m2 = mk().fit(X[m], y[m], sample_weight=compute_sample_weight("balanced", y[m])); P2 = pd.DataFrame(m2.predict_proba(Xt), columns=m2.classes_)   # sklearn 1.1.3 호환: HGB 에 class_weight 없음
    out = pd.DataFrame({"IDLE_HOVER": P1["IDLE_HOVER"], "HMSL": P1["HMSL"], "TURN": P1["TURN"], "ASCEND": P1["AD"] * P2["ASCEND"], "DESCEND": P1["AD"] * P2["DESCEND"]})
    return out[CL].values

def ens_probs(X, y, Xt): return (probs_A(rf, X, y, Xt) + probs_A(hgb, X, y, Xt)) / 2

Ftr, Fte = make_features(tr), make_features(te)
print("특징 수:", Ftr.shape[1], "| train/test 행:", len(tr), len(te))

# ---- 비행(uid) 단위 Leave-One-Flight-Out 검증 ----
P = np.zeros((len(tr), 5))
for u in tr.uid.unique():
    m = (tr.uid == u).values; P[m] = ens_probs(Ftr[~m].values, tr.flight_state[~m], Ftr[m].values)
y = tr.flight_state.values
for name, w in (("가중 없음", np.ones(5)), ("ASCEND x3, DESCEND x5 (낙관적)", W)):
    pred = np.array(CL)[(P * w).argmax(1)]; f = f1_score(y, pred, labels=CL, average=None, zero_division=0)
    print(f"{name:32s} 정확도 {np.mean(pred == y):.4f} | Macro F1 {f.mean():.4f} | " + " ".join(f"{c[:4]}:{v:.3f}" for c, v in zip(CL, f)))

# ---- 전체 train 으로 재학습해 test 예측 ----
Pt = ens_probs(Ftr.values, tr.flight_state, Fte.values)
pred_te = pd.DataFrame({"id": te.id, "flight_state": np.array(CL)[(Pt * W).argmax(1)]})
submission = sample[["id"]].merge(pred_te, on="id", how="left")
assert submission.flight_state.notna().all() and set(submission.flight_state) <= set(CL) and len(submission) == len(sample)
submission.to_csv("submission.csv", index=False)
print("예측 분포:", submission.flight_state.value_counts(normalize=True).round(3).to_dict()); submission.head()''')

nb.cells = nb.cells[:2] + [md, code] + nb.cells[3:]
for c in nb.cells:
    if c.cell_type == "code": c.outputs = []; c.execution_count = None
NotebookClient(nb, timeout=3000, kernel_name="python3", resources={"metadata": {"path": os.getcwd()}}).execute()
nbformat.write(nb, "work.ipynb")
