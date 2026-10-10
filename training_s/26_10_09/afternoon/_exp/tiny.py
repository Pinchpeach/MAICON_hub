"""작은 사람 박스의 conf 분포: 2번 train 30장, 관점별 원시 예측(conf>=0.03)과 융합 군집 기준."""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "_exp")
from lib import _iou_inter
from PIL import Image, ImageDraw
p = "problem02"; tl = pd.read_csv(f"{p}/data/train_labels.csv"); V = pickle.load(open(f"_exp/{p}_views.pkl", "rb")); KEYS = ["full960", "full1280", "tile960"]
def clusters(views):
    B, S, Vw = [], [], []
    for vi, k in enumerate(KEYS):
        b, s, c = views[k]; m = c == 0; B.append(b[m]); S.append(s[m]); Vw.append(np.full(int(m.sum()), vi))
    B, S, Vw = map(np.concatenate, (B, S, Vw)); reps, mem = [], []
    for i in np.argsort(-S):
        if reps:
            io, ct = _iou_inter(B[i], B[reps]); j = np.where((io >= 0.5) | (ct >= 0.7))[0]
            if len(j): mem[j[0]].append(i); continue
        reps.append(i); mem.append([i])
    out = []
    for r, m in zip(reps, mem):
        best = {}
        for i in m: best[Vw[i]] = max(best.get(Vw[i], 0), S[i])
        out.append((B[r], S[r], sum(best.values()) / 3, len(best)))      # 박스, 최고 conf, 융합 점수, 본 관점 수
    return out
rows = []
for f, y in zip(tl.filename, tl.person_count):
    W, H = Image.open(f"{p}/data/train_images/{f}").size
    for b, mx, sc, nv in clusters(V[("train_images", f)]):
        rows.append(dict(f=f, y=y, h_px=(b[3] - b[1]) * H, w_px=(b[2] - b[0]) * W, hfrac=b[3] - b[1], maxconf=mx, score=sc, nviews=nv, box=b))
D = pd.DataFrame(rows); D["size"] = pd.cut(D.h_px, [0, 20, 40, 80, 160, 1000], labels=["~20px", "20~40px", "40~80px", "80~160px", "160px~"])
pd.set_option("display.width", 200)
print("군집 수(최고 conf 0.03 이상) 크기별 x 최고 conf 구간:")
D["confbin"] = pd.cut(D.maxconf, [0, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 1.0])
print(pd.crosstab(D["size"], D.confbin).to_string())
print("\n크기별 최고 conf 분위수(전체 후보, 오탐 포함):"); print(D.groupby("size", observed=True).maxconf.describe()[["count", "25%", "50%", "75%", "max"]].round(3).to_string())
# 현재 규칙(2관점 이상, 점수>=0.15)으로 세어지는 것과 아닌 것
D["counted"] = (D.nviews >= 2) & (D.score >= 0.15)
print("\n현재 규칙으로 세어진 비율(크기별):"); print(D.groupby("size", observed=True).counted.agg(["sum", "count", "mean"]).round(2).to_string())
# 부족분과 '작고 낮은 conf' 후보 수 비교
g = D.groupby("f").agg(y=("y", "first"), counted=("counted", "sum")).reindex(tl.filename); g["deficit"] = g.y - g.counted
small_lo = D[(~D.counted) & (D.h_px < 40)].groupby("f").size().reindex(tl.filename).fillna(0)
print("\n덜 센 이미지(부족분>=2):"); print(pd.DataFrame({"정답": g.y, "센 수": g.counted, "부족": g.deficit, "안 세어진 40px 미만 후보": small_lo})[g.deficit >= 2].to_string())
pickle.dump(D, open("_exp/p02_tiny.pkl", "wb"))
