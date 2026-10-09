"""이미지 크기(= 출처 단서)별 클래스 사전확률로 클래스 점수를 보정: 사전확률은 train 에서만 추정, val 로 평가(정직)."""
import pickle, glob, os, numpy as np, collections
from PIL import Image
from p1_ens import wbf, gt_counts
raw, rawf = pickle.load(open('audit/_val_raw.pkl', 'rb'))
vp = sorted(glob.glob('data/val/images/*')); G = np.array([gt_counts(p) for p in vp])
size = lambda p: '%dx%d' % Image.open(p).size
tp = sorted(glob.glob('data/train/images/*')); tg = np.array([gt_counts(p) for p in tp]); ts = [size(p) for p in tp]
vs = [size(p) for p in vp]
glob_c = tg.sum(0) + 1.0; glob_p = glob_c / glob_c.sum()
def prior_w(s, alpha, min_boxes):
    idx = [i for i, x in enumerate(ts) if x == s]; c = tg[idx].sum(0)
    if c.sum() < min_boxes: return np.ones(4)
    p = (c + alpha * glob_p * 4) / (c.sum() + alpha * 4)      # 스무딩된 크기별 클래스 비율
    return p / glob_p
def cls_arrays(keys, flip):
    src = rawf if flip else raw; n = len(keys) * (2 if flip else 1)
    return [np.array([c[3] for c in wbf([s for k in keys for s in src[k][i]], n, agnostic=True)]).reshape(-1, 4) for i in range(len(vp))]
def counts(arr, w, thr):
    c = np.zeros(4, int)
    if len(arr) == 0: return c
    sc = arr * w; k = sc.argmax(1); ok = arr.sum(1) >= thr          # 존재 여부는 원래 점수 합, 클래스만 보정
    for j in range(4): c[j] = int(((k == j) & ok).sum())
    return c
TH = (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55); T2 = TH[1:6]
for name, keys, flip in [('a100+flip', ['a100'], True), ('b0+b1+a100', ['b0', 'b1', 'a100'], False), ('25ep6+a100', ['b0', 'b1', 'k0', 'k1', 's0', 's1', 'a100'], False)]:
    A = cls_arrays(keys, flip)
    def acc(alpha, mb, t):
        if alpha is None: W = [np.ones(4)] * len(vp)
        else: W = [prior_w(s, alpha, mb) for s in vs]
        return np.mean([(counts(A[i], W[i], t) == G[i]).all() for i in range(len(vp))])
    base = [acc(None, 0, t) for t in TH]
    print(f'{name:12s} 기준: 평균 {np.mean(base[1:6]):.4f} 최고 {max(base):.4f}')
    for alpha in (0.5, 1, 2, 5):
        for mb in (10, 20):
            r = [acc(alpha, mb, t) for t in TH]; print(f'   prior alpha={alpha} min_boxes={mb}: 평균(0.3-0.5) {np.mean(r[1:6]):.4f} 최고 {max(r):.4f}')
