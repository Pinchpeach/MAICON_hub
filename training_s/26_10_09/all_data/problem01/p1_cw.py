"""클래스 가중 + 임계값 후처리의 정직한 이득 추정: val 을 반으로 나눠 한쪽에서 튜닝, 다른 쪽에서 평가(30회 반복)."""
import pickle, glob, itertools, numpy as np
from p1_ens import wbf, gt_counts
raw, rawf = pickle.load(open('audit/_val_raw.pkl', 'rb'))
paths = sorted(glob.glob('data/val/images/*')); G = np.array([gt_counts(p) for p in paths])
def cls_arrays(keys, flip):
    src = rawf if flip else raw; n = len(keys) * (2 if flip else 1)
    out = []
    for i in range(len(paths)):
        cl = wbf([s for k in keys for s in src[k][i]], n, agnostic=True)
        out.append(np.array([c[3] for c in cl]).reshape(-1, 4))
    return out
def counts(arr, w, thr):
    c = np.zeros(4, int)
    if len(arr) == 0: return c
    sc = arr * w; k = sc.argmax(1); ok = sc.max(1) >= thr
    for j in range(4): c[j] = int(((k == j) & ok).sum())
    return c
W1 = (1,); WT = (1, 1.5, 2, 3); WM = (1, 1.3, 1.6, 2); WC = (1, 1.3, 1.6); TH = (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55)
grid = [((1, a, b, c), t) for a in WT for b in WM for c in WC for t in TH]
def acc(A, idx, w, t): return np.mean([(counts(A[i], np.array(w), t) == G[i]).all() for i in idx])
rng = np.random.default_rng(0)
for name, keys, flip in [("a100+flip", ["a100"], True), ("b0+b1+a100", ["b0", "b1", "a100"], False), ("25ep6+a100", ["b0", "b1", "k0", "k1", "s0", "s1", "a100"], False)]:
    A = cls_arrays(keys, flip)
    base_t = {t: acc(A, range(160), (1, 1, 1, 1), t) for t in TH}
    gains = []; base = []; tuned = []
    for r in range(30):
        perm = rng.permutation(160); a, b = perm[:80], perm[80:]
        for tr, te in ((a, b), (b, a)):
            bt = max(TH, key=lambda t: acc(A, tr, (1, 1, 1, 1), t))      # 기준: 전역 임계값만 튜닝
            bw = max(grid, key=lambda g: acc(A, tr, g[0], g[1]))          # 클래스 가중 + 임계값 튜닝
            base.append(acc(A, te, (1, 1, 1, 1), bt)); tuned.append(acc(A, te, bw[0], bw[1]))
    full = max(grid, key=lambda g: acc(A, range(160), g[0], g[1]))
    print(f"{name:12s} 전역 임계값만 {np.mean(base):.4f} | 클래스 가중+임계값(반으로 튜닝→나머지 평가) {np.mean(tuned):.4f} | 전체 val 로 튜닝한 최고(낙관적) {acc(A, range(160), full[0], full[1]):.4f} w={full[0]} thr={full[1]}", flush=True)
