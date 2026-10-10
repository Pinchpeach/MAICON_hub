"""v3 에서 종류는 맞았지만 수량이 틀린 이미지의 진단(답지는 진단용)과 시트 생성."""
import os, pickle, numpy as np, pandas as pd, torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
AIR, BIRD = 4, 14; AO = "data/aerial_objects"; cols = ["airplane", "bird", "drone"]; dev = 0 if torch.cuda.is_available() else "cpu"
S = pd.read_csv("submission.csv").set_index("filename"); A = pd.read_csv(r"C:\Users\User\Downloads\submission (1).csv").set_index("filename")
PB = pickle.load(open("_final_preds.pkl", "rb"))["P_base"]; W = "work_p2/runs/drone_s_neg/weights/last.pt"; fn = "_v3_drone_pred.pkl"
if os.path.exists(fn): PD = pickle.load(open(fn, "rb"))
else:
    m = YOLO(W); fs = sorted(PB); PD = {}
    for i in range(0, 300, 10):
        for f, r in zip(fs[i:i + 10], m.predict([os.path.join(AO, x) for x in fs[i:i + 10]], imgsz=640, conf=0.05, iou=0.6, device=dev, verbose=False)): PD[f] = (r.boxes.conf.cpu().numpy(), r.boxes.xyxyn.cpu().numpy())
    pickle.dump(PD, open(fn, "wb"))
rows = []
for f in S.index:
    ka = cols[int(np.argmax(A.loc[f, cols].values))]; ks = cols[int(np.argmax(S.loc[f, cols].values))]
    if ka != ks or (A.loc[f, cols].values > 0).sum() != 1: continue
    t, p = int(A.loc[f, ka]), int(S.loc[f, ka]); W_, H_ = Image.open(os.path.join(AO, f)).size
    rows.append(dict(f=f, kind=ka, true=t, pred=p, diff=p - t, mp=W_ * H_ / 1e6))
D = pd.DataFrame(rows); E = D[D["diff"] != 0]
print("종류가 맞은 이미지 %d장 중 수량 오류 %d장" % (len(D), len(E)))
for k in cols:
    d = D[D.kind == k]; e = d[d["diff"] != 0]
    print("  %-9s %3d장: 오류 %2d (과소 %2d, 과다 %2d) | 정답 수량별 정확도: %s" % (k, len(d), len(e), (e["diff"] < 0).sum(), (e["diff"] > 0).sum(), {int(t): "%d/%d" % ((g["diff"] == 0).sum(), len(g)) for t, g in d.groupby("true")}))
print("  오차 크기 분포:", E["diff"].value_counts().sort_index().to_dict())
print("  정답 1개인 이미지 %d장 중 오류 %d장 | 정답 2개 이상 %d장 중 오류 %d장" % ((D.true == 1).sum(), ((D.true == 1) & (D["diff"] != 0)).sum(), (D.true >= 2).sum(), ((D.true >= 2) & (D["diff"] != 0)).sum()))
E.to_csv("_count_errors.csv", index=False)
ft = lambda s: ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", s); E = E.sort_values(["kind", "diff"]); cell = (330, 270); ncol = 7; nrow = (len(E) + ncol - 1) // ncol
Sh = Image.new("RGB", (ncol * cell[0], nrow * cell[1] + 30), (252, 252, 251)); ImageDraw.Draw(Sh).text((8, 6), "v3 수량 오류 %d장 (파랑 = 비행기, 초록 = 새, 주황 = 드론 박스; 굵은 선 = 센 박스, 가는 선 = 임계값 아래 후보)" % len(E), fill=(11, 11, 11), font=ft(15))
for i, r in enumerate(E.itertuples()):
    im = Image.open(os.path.join(AO, r.f)).convert("RGB"); w, h = im.size; q = min((cell[0] - 6) / w, (cell[1] - 24) / h); t = im.resize((int(w * q), int(h * q)), Image.LANCZOS); dr = ImageDraw.Draw(t)
    if r.kind == "drone": bx, sc, col, thr = PD[r.f][1], PD[r.f][0], (235, 104, 52), 0.35
    else:
        c, s, b = PB[r.f]; k = AIR if r.kind == "airplane" else BIRD; bx, sc, col, thr = b[c == k], s[c == k], ((42, 120, 214) if k == AIR else (27, 175, 122)), 0.25
    for b_, s_ in zip(bx, sc):
        if s_ < 0.08: continue
        dr.rectangle([b_[0] * t.size[0], b_[1] * t.size[1], b_[2] * t.size[0], b_[3] * t.size[1]], outline=col, width=3 if s_ >= thr else 1); dr.text((b_[0] * t.size[0] + 2, b_[1] * t.size[1] + 1), "%.2f" % s_, fill=col, font=ft(11))
    x, y = (i % ncol) * cell[0], 30 + (i // ncol) * cell[1]; Sh.paste(t, (x + 3, y + 20)); ImageDraw.Draw(Sh).text((x + 4, y + 2), "%s %s 정답 %d → 제출 %d (%.1fMP)" % (r.f[4:7], r.kind[:4], r.true, r.pred, r.mp), fill=(11, 11, 11), font=ft(12))
Sh.save("_viz/count_errors_v3.jpg", quality=88); print("sheet", Sh.size)
