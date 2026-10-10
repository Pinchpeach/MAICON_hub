"""비행기와 드론을 가를 단서가 있는가: 정답 드론 박스 대 테스트에서 기본 모델이 비행기로 본 박스의 모양(가로세로비, 면적 비율)."""
import os, pickle, numpy as np, pandas as pd
from PIL import Image
TL, AO = "data/transfer_learning", "data/aerial_objects"; AIR = 4
rows = []
for f in sorted(x for x in os.listdir(TL) if x.lower().endswith((".jpg", ".png"))):
    W, H = Image.open(os.path.join(TL, f)).size; x, y, w, h = map(float, open(os.path.join(TL, os.path.splitext(f)[0] + ".txt")).read().split()[1:5]); rows.append(dict(kind="드론(정답)", ar=w * W / (h * H), area=w * h, rel=max(w * W, h * H) / max(W, H)))
G = pd.DataFrame(rows)
P = pickle.load(open("_final_preds.pkl", "rb"))["P_base"]; S = pd.read_csv("_rule_scores.csv").set_index("f"); rows = []
for f, (c, s, b) in P.items():
    m = (c == AIR) & (s >= 0.5)
    if not m.any(): continue
    i = np.argmax(np.where(m, s, -1)); W, H = Image.open(os.path.join(AO, f)).size; x0, y0, x1, y1 = b[i]; w, h = x1 - x0, y1 - y0
    rows.append(dict(f=f, kind="테스트 비행기로 본 것", ar=w * W / max(h * H, 1), area=w * h, rel=max(w * W, h * H) / max(W, H), sd=S.loc[f, "sd"], sa=S.loc[f, "sa"], n=int(m.sum())))
A = pd.DataFrame(rows)
print("== 박스 모양 비교 (가로/세로, 면적 비율, 긴 변/이미지 긴 변) ==")
for nm, d in (("정답 드론 58장", G), ("테스트에서 비행기(점수 0.5 이상)로 본 이미지 %d장" % len(A), A)):
    print("  %s" % nm); print("    가로/세로 분위(10,25,50,75,90): %s" % np.percentile(d.ar, [10, 25, 50, 75, 90]).round(2).tolist()); print("    면적 비율 분위: %s" % np.percentile(d.area, [10, 25, 50, 75, 90]).round(2).tolist())
for t in (1.5, 2.0, 2.5, 3.0): print("  가로/세로 >= %.1f: 드론 %2d/%d (%.0f%%) | 테스트 비행기로 본 것 %3d/%d (%.0f%%)" % (t, (G.ar >= t).sum(), len(G), 100 * (G.ar >= t).mean(), (A.ar >= t).sum(), len(A), 100 * (A.ar >= t).mean()))
print("  가로/세로 < 1.3 (정사각형에 가까움): 드론 %d/%d | 테스트 %d/%d" % ((G.ar < 1.3).sum(), len(G), (A.ar < 1.3).sum(), len(A)))
print("\n== 테스트에서 비행기로 본 이미지의 분류 후보 (가로/세로 별) ==")
print(pd.crosstab(pd.cut(A.ar, [0, 1.3, 1.8, 2.5, 10]), pd.cut(A.area, [0, .1, .3, .6, 1.01])).to_string())
A.to_csv("_airplane_boxes.csv", index=False)
