"""검토용 그림: 후보 박스에 번호를 붙이고 10% 눈금을 그린다. 후보 목록은 review/candidates.json 에 저장"""
import sys, os, json; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
from harness import labels, DATA, ROOT
import make_dummy as M
OUT = os.path.join(ROOT, "_exp2", "review"); os.makedirs(OUT, exist_ok=True)
S = pd.read_csv(os.path.join(ROOT, "_exp2", "dummy_boxes", "summary.csv")); todo = sys.argv[1:] or list(S.filename[~S.both_match]); C = {}
for f in todo:
    n = list(labels.filename).index(f); img = Image.open(os.path.join(DATA, "train_images", f)).convert("RGB"); W, H = img.size; sc = 1000 / max(W, H); can = img.resize((int(W * sc), int(H * sc)), Image.LANCZOS); cw, ch = can.size
    pad = 26; full = Image.new("RGB", (cw + pad, ch + pad + 24), (252, 252, 251)); full.paste(can, (pad, pad + 24)); dr = ImageDraw.Draw(full); ft = M.font(15); fs = M.font(12)
    for i in range(11):                                              # 10% 눈금 (0~100)
        x = pad + cw * i / 10; y = pad + 24 + ch * i / 10; dr.line([x, pad + 24, x, pad + 24 + ch], fill=(255, 255, 255), width=1); dr.line([pad, y, pad + cw, y], fill=(255, 255, 255), width=1)
        dr.text((x - 8, 26), str(i * 10), fill=(11, 11, 11), font=fs); dr.text((2, y - 7), str(i * 10), fill=(11, 11, 11), font=fs)
    C[f] = []
    for g, pre in (("person", "P"), ("vehicle", "V")):
        cl = sorted([c for c in M.clusters(f, g) if c[1] >= 0.03], key=lambda c: -c[1])
        for k, (b, s, kk, nv) in enumerate(cl, 1):
            kept = s >= M.CFG[g]["thr"]; col = M.COL[g] if kept else ((120, 160, 215) if g == "person" else (205, 150, 120)); cid = f"{pre}{k}"
            x0, y0, x1, y1 = pad + b[0] * cw, pad + 24 + b[1] * ch, pad + b[2] * cw, pad + 24 + b[3] * ch; dr.rectangle([x0, y0, x1, y1], outline=col, width=3 if kept else 1)
            tag = f"{cid} {s:.2f}"; tb = dr.textbbox((0, 0), tag, font=ft); dr.rectangle([x0, y0, x0 + tb[2] + 6, y0 + tb[3] + 5], fill=col); dr.text((x0 + 3, y0 + 1), tag, fill=(255, 255, 255), font=ft)
            C[f].append(dict(id=cid, group=g, box=[round(float(v), 4) for v in b], score=round(float(s), 3), kept=bool(kept)))
    dr.text((pad, 4), f"{f} ({W}x{H})  정답 사람 {int(labels.person_count[n])} · 차량 {int(labels.vehicle_count[n])} | 굵은 선 = 센 박스, 가는 선 = 임계값 아래 후보 | 눈금은 % 좌표", fill=(11, 11, 11), font=M.font(14))
    full.save(os.path.join(OUT, f), quality=90)
old = json.load(open(os.path.join(OUT, "candidates.json"))) if os.path.exists(os.path.join(OUT, "candidates.json")) else {}; old.update(C); json.dump(old, open(os.path.join(OUT, "candidates.json"), "w"), indent=0)
print("검토 대상", len(todo), todo)
