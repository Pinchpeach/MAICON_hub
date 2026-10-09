import pandas as pd, numpy as np, warnings, time, sys
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix

CL = ["IDLE_HOVER", "ASCEND", "DESCEND", "TURN", "HMSL"]
tr = pd.read_csv("data/train.csv").sort_values(["uid", "seq"]).reset_index(drop=True)


def make_features(df, raw=False):
    """uid 안에서 현재+과거 행만 쓰는 특징 (미래 값 미사용)."""
    g = df.groupby("uid", sort=False)
    F = pd.DataFrame(index=df.index)
    ax, ay, az = df.angular_x.abs(), df.angular_y.abs(), df.angular_z.abs()
    F["abs_ang_x"], F["abs_ang_y"], F["abs_ang_z"] = ax, ay, az
    ang = np.sqrt(df.angular_x ** 2 + df.angular_y ** 2 + df.angular_z ** 2)
    acc = np.sqrt(df.linear_acceleration_x ** 2 + df.linear_acceleration_y ** 2 + df.linear_acceleration_z ** 2)
    F["ang_norm"], F["acc_norm"] = ang, acc
    for c in ("linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"):
        F["abs_" + c] = df[c].abs()
    uid = df.uid

    def roll(s, n, fn):
        r = s.groupby(uid, sort=False).rolling(n, min_periods=2)
        return getattr(r, fn)().reset_index(level=0, drop=True).sort_index()

    for n in (5, 10, 50):
        F[f"ang_norm_m{n}"] = roll(ang, n, "mean"); F[f"acc_norm_m{n}"] = roll(acc, n, "mean")
        F[f"abs_ang_z_m{n}"] = roll(az, n, "mean")
        F[f"acc_z_m{n}"] = roll(df.linear_acceleration_z, n, "mean")          # 부호 있는 z 가속도 평균
    for n in (10, 50):
        F[f"ang_norm_s{n}"] = roll(ang, n, "std"); F[f"acc_norm_s{n}"] = roll(acc, n, "std")
        F[f"acc_z_s{n}"] = roll(df.linear_acceleration_z, n, "std")
    F["ang_z_s10"] = roll(df.angular_z, 10, "std")
    for c, k in (("linear_acceleration_x", "x"), ("linear_acceleration_y", "y")):
        F[f"acc_{k}_s10"] = roll(df[c], 10, "std")
    for n in (10, 50):
        F[f"cur_slope{n}"] = (df.battery_current - g.battery_current.shift(n)) / n
    F["volt_slope50"] = (df.battery_voltage - g.battery_voltage.shift(50)) / 50
    F["volt_rel"] = df.battery_voltage - g.battery_voltage.transform("first")      # 비행 시작 대비
    F["cur_m50"] = roll(df.battery_current, 50, "mean")
    if raw:
        for c in ("battery_voltage", "battery_current", "wind_speed", "wind_angle"): F["raw_" + c] = df[c]
    return F.fillna(0.0)


def rf(**k): return RandomForestClassifier(300, min_samples_leaf=3, n_jobs=-1, random_state=0, **k)
def hgb(**k): return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, random_state=0, **k)


def fit_predict(kind, X, y, Xt, mk=rf):
    y = y.values
    if kind == "flat":
        return mk().fit(X, y).predict(Xt)
    if kind == "A":     # 1단계: IDLE/HMSL/TURN/(ASC+DESC 묶음) -> 2단계: 묶음 안에서 ASC vs DESC
        y1 = np.where(np.isin(y, ["ASCEND", "DESCEND"]), "AD", y)
        p = mk().fit(X, y1).predict(Xt).astype(object)
        m = np.isin(y, ["ASCEND", "DESCEND"])
        s2 = mk(class_weight="balanced").fit(X[m], y[m])
        sel = p == "AD"
        if sel.any(): p[sel] = s2.predict(Xt[sel])
        return p
    if kind.startswith("B"):  # 0단계: DESCEND vs 나머지 전용 이진 모델 -> 나머지 4클래스
        thr = float(kind[1:]) if len(kind) > 1 else 0.5
        yb = (y == "DESCEND")
        pd_ = mk(class_weight="balanced").fit(X, yb).predict_proba(Xt)[:, 1]
        m = ~yb
        p = mk().fit(X[m], y[m]).predict(Xt).astype(object)
        p[pd_ >= thr] = "DESCEND"
        return p


def run(name, kind, feats, mk=rf):
    t = time.time(); oof = np.empty(len(tr), dtype=object)
    for u in tr.uid.unique():
        te = (tr.uid == u).values
        oof[te] = fit_predict(kind, feats[~te].values, tr.flight_state[~te], feats[te].values, mk)
    yt = tr.flight_state.values
    f = f1_score(yt, oof, labels=CL, average=None, zero_division=0)
    print(f"{name:34s} macroF1 {f.mean():.4f} | " + " ".join(f"{c[:4]}:{v:.3f}" for c, v in zip(CL, f)) + f" | {time.time() - t:.0f}s", flush=True)
    return oof


if __name__ == "__main__":
    Fb, Fr = make_features(tr), make_features(tr, raw=True)
    print("특징 수 base", Fb.shape[1], "raw 포함", Fr.shape[1])
    print("-- 평탄(flat) 5클래스 --")
    run("flat RF base", "flat", Fb); run("flat RF +raw", "flat", Fr)
    run("flat HGB base", "flat", Fb, hgb)
    print("-- 계층 A: (ASC+DESC 묶음) -> 2단계 --")
    run("A RF base", "A", Fb); run("A RF +raw", "A", Fr); run("A HGB base", "A", Fb, hgb)
    print("-- 계층 B: DESCEND 전용 0단계 -> 나머지 --")
    for k in ("B0.5", "B0.7"): run(f"{k} RF base", k, Fb)
    run("B0.5 RF +raw", "B0.5", Fr)
