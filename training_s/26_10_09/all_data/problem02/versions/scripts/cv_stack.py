import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from cv2 import make_features_v2, tr, CL

OUT = os.environ.get("STACK_OUT", "stack_cache"); os.makedirs(OUT, exist_ok=True)
F = make_features_v2(tr).values
y = tr.flight_state.values; uid = tr.uid.values; flights = list(pd.unique(uid))


def level1(Xa, ya, Xb):
    """상태 5개 x (RF, HGB) one-vs-rest 이진 모델 -> Xb 에 대한 확률 (n, 10). 열 순서: CL 순 RF 5개, HGB 5개."""
    out = []
    for kind in os.environ.get("STACK_KINDS", "rf,hgb").split(","):
        for s in CL:
            yb = (ya == s)
            if yb.sum() < 5: out.append(np.zeros(len(Xb))); continue            # 이 학습 비행들에 상태가 거의 없으면 0
            if kind == "rf":
                m = RandomForestClassifier(200, min_samples_leaf=3, n_jobs=-1, random_state=0, class_weight="balanced_subsample").fit(Xa, yb)
            else:
                m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_leaf_nodes=15, random_state=0).fit(Xa, yb, sample_weight=compute_sample_weight("balanced", yb))
            out.append(m.predict_proba(Xb)[:, 1])
    return np.column_stack(out)   # kinds 수 x 5 열


if __name__ == "__main__":
    t0 = time.time()
    for o in flights:                                        # 바깥: 비행 하나를 통째로 검증용으로
        fn = f"{OUT}/outer_{o}.npz"
        if os.path.exists(fn): continue
        tm = uid != o; vm = ~tm
        inner = np.zeros((tm.sum(), 5 * len(os.environ.get("STACK_KINDS", "rf,hgb").split(",")))); Xt, yt, ut = F[tm], y[tm], uid[tm]
        for i in pd.unique(ut):                              # 안쪽: 학습 비행들 사이의 LOFO OOF
            m = ut == i
            inner[m] = level1(Xt[~m], yt[~m], Xt[m])
        val = level1(Xt, yt, F[vm])                          # 학습 비행 전체로 stack1 -> 검증 비행
        np.savez(fn, inner=inner, val=val, tm=tm)
        print(f"outer {o} 완료 {time.time() - t0:.0f}s", flush=True)
    print("ALL DONE", flush=True)
