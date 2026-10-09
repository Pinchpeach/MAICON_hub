import pandas as pd, numpy as np, warnings, time
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from cv_hier import make_features, tr, CL


def make_features_v2(df):
    F = make_features(df)
    uid = df.uid
    ang = np.sqrt(df.angular_x ** 2 + df.angular_y ** 2 + df.angular_z ** 2)
    acc = np.sqrt(df.linear_acceleration_x ** 2 + df.linear_acceleration_y ** 2 + df.linear_acceleration_z ** 2)
    az = df.linear_acceleration_z
    g = df.groupby("uid", sort=False)

    def roll(s, n, fn):
        return getattr(s.groupby(uid, sort=False).rolling(n, min_periods=2), fn)().reset_index(level=0, drop=True).sort_index()

    def ewm(s, span): return s.groupby(uid, sort=False).transform(lambda x: x.ewm(span=span, adjust=False).mean())
    for n in (100, 200):
        F[f"acc_norm_m{n}"] = roll(acc, n, "mean"); F[f"ang_norm_m{n}"] = roll(ang, n, "mean"); F[f"acc_z_m{n}"] = roll(az, n, "mean")
    for sp in (5, 20, 100):
        F[f"acc_norm_e{sp}"] = ewm(acc, sp); F[f"ang_norm_e{sp}"] = ewm(ang, sp); F[f"acc_z_e{sp}"] = ewm(az, sp)
    for n in (10, 50):
        F[f"acc_norm_max{n}"] = roll(acc, n, "max"); F[f"acc_z_max{n}"] = roll(az, n, "max"); F[f"acc_z_min{n}"] = roll(az, n, "min")
    F["acc_trend"] = F["acc_norm_m10"] - F["acc_norm_m50"]; F["ang_trend"] = F["ang_norm_m10"] - F["ang_norm_m50"]
    F["cur_slope100"] = (df.battery_current - g.battery_current.shift(100)) / 100
    F["volt_slope100"] = (df.battery_voltage - g.battery_voltage.shift(100)) / 100
    return F.fillna(0.0)


def rf(**k): return RandomForestClassifier(300, min_samples_leaf=3, n_jobs=-1, random_state=0, **k)
def hgb(**k): return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, random_state=0, **k)


def probs_A(mk, X, y, Xt):
    """A 구조의 확률: P(IDLE,HMSL,TURN,AD) x P(ASC|AD), P(DESC|AD) -> CL 순서 5열"""
    y = np.asarray(y); y1 = np.where(np.isin(y, ["ASCEND", "DESCEND"]), "AD", y)
    m1 = mk().fit(X, y1); P1 = pd.DataFrame(m1.predict_proba(Xt), columns=m1.classes_)
    m = np.isin(y, ["ASCEND", "DESCEND"]); m2 = mk(class_weight="balanced").fit(X[m], y[m]); P2 = pd.DataFrame(m2.predict_proba(Xt), columns=m2.classes_)
    out = pd.DataFrame({"IDLE_HOVER": P1["IDLE_HOVER"], "HMSL": P1["HMSL"], "TURN": P1["TURN"], "ASCEND": P1["AD"] * P2["ASCEND"], "DESCEND": P1["AD"] * P2["DESCEND"]})
    return out[CL].values


def oof_probs(F, mk):
    P = np.zeros((len(tr), 5))
    for u in tr.uid.unique():
        te = (tr.uid == u).values
        P[te] = probs_A(mk, F[~te].values, tr.flight_state[~te], F[te].values)
    return P


def smooth(P, uid, alpha):
    """과거 행만 쓰는 지수 평활: S_t = alpha*S_{t-1} + (1-alpha)*P_t"""
    S = np.zeros_like(P)
    for u in np.unique(uid):
        idx = np.flatnonzero(uid == u); s = P[idx[0]].copy(); S[idx[0]] = s
        for i in idx[1:]:
            s = alpha * s + (1 - alpha) * P[i]; S[i] = s
    return S


def report(name, P, w=None):
    Q = P * (w if w is not None else 1); pred = np.array(CL)[Q.argmax(1)]; y = tr.flight_state.values
    f = f1_score(y, pred, labels=CL, average=None, zero_division=0)
    print(f"{name:40s} acc {np.mean(pred == y):.4f} macroF1 {f.mean():.4f} | " + " ".join(f"{c[:4]}:{v:.3f}" for c, v in zip(CL, f)), flush=True)
    return pred


if __name__ == "__main__":
    F1, F2 = make_features(tr), make_features_v2(tr)
    print("특징 수 v1", F1.shape[1], "v2", F2.shape[1])
    t = time.time()
    P1r = oof_probs(F1, rf); report("v1 A-RF", P1r)
    P2r = oof_probs(F2, rf); report("v2 A-RF (후보1: 특징확장)", P2r)
    P2h = oof_probs(F2, hgb); report("v2 A-HGB", P2h)
    Pe = (P2r + P2h) / 2; report("v2 A-RF+HGB 확률평균 (후보2)", Pe)
    print(f"({time.time() - t:.0f}s)")
    uid = tr.uid.values
    for a in (0.5, 0.8, 0.9):
        report(f"  + 과거 지수평활 alpha={a}", smooth(Pe, uid, a))
    np.save("_Pe.npy", Pe); np.save("_P2r.npy", P2r); np.save("_P2h.npy", P2h)
