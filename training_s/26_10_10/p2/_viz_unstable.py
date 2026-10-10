import os, pickle, numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
R = pickle.load(open("_p2_coco_pass.pkl", "rb")); U = pd.read_csv("_p2_unstable.csv"); AO = "data/aerial_objects"
f_ = lambda s: ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", s)
sel = list(U[U.type == "kind"].f) + ["test064.jpg", "test100.jpg", "test214.jpg", "test236.jpg"] + list(U[U.type == "count"].f)[:11]; sel = list(dict.fromkeys(sel))[:28]
cell = (300, 250); cols = 7; rows = (len(sel) + cols - 1) // cols; S = Image.new("RGB", (cols * cell[0], rows * cell[1] + 30), (252, 252, 251)); ImageDraw.Draw(S).text((8, 6), "임계값에 따라 종류·수량이 바뀌는 이미지 (파랑 = airplane, 초록 = bird, 회색 = 그 밖의 클래스, 숫자 = 점수)", fill=(11, 11, 11), font=f_(15))
for i, f in enumerate(sel):
    im = Image.open(os.path.join(AO, f)).convert("RGB"); W, H = im.size; r = min((cell[0] - 6) / W, (cell[1] - 22) / H); t = im.resize((int(W * r), int(H * r)), Image.LANCZOS); dr = ImageDraw.Draw(t); c, s, b = R[f]
    for k, sc, bx in zip(c, s, b):
        if sc < 0.1: continue
        col = (42, 120, 214) if k == 4 else ((27, 175, 122) if k == 14 else (150, 150, 146))
        dr.rectangle([bx[0] * t.size[0], bx[1] * t.size[1], bx[2] * t.size[0], bx[3] * t.size[1]], outline=col, width=2); dr.text((bx[0] * t.size[0] + 2, bx[1] * t.size[1] + 1), f"{sc:.2f}", fill=col, font=f_(12))
    x, y = (i % cols) * cell[0], 30 + (i // cols) * cell[1]; S.paste(t, (x + 3, y + 20)); ImageDraw.Draw(S).text((x + 4, y + 2), f[:-4] + (" (종류)" if f in set(U[U.type == "kind"].f) else ""), fill=(11, 11, 11), font=f_(12))
S.save("_viz/unstable_images.jpg", quality=88); print(len(sel))
