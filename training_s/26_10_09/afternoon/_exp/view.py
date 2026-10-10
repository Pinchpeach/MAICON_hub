import sys, pickle, numpy as np, pandas as pd
from PIL import Image, ImageDraw
p, sz, iou, th, tgt, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5], sys.argv[6]
G = {"person": [0], "vehicle": [2, 5, 7]}; tl = pd.read_csv(f"{p}/data/train_labels.csv"); R = pickle.load(open(f"_exp/{p}_raw.pkl", "rb"))[(sz, iou)]
y = tl[f"{tgt}_count"].values; pr = np.array([int((np.isin(c, G[tgt]) & (s >= th)).sum()) for c, s, _ in R]); order = np.argsort(-np.abs(pr - y))[:12]
print("오차 분포", dict(zip(*np.unique(pr - y, return_counts=True))))
S = 440; sheet = Image.new("RGB", (4 * S, 3 * S), (30, 30, 30))
for j, i in enumerate(order):
    im = Image.open(f"{p}/data/train_images/{tl.filename[i]}").convert("RGB"); W, H = im.size; d = ImageDraw.Draw(im); c, s, b = R[i]
    for k in range(len(c)):
        if c[k] in G[tgt] and s[k] >= 0.1:
            col = (0, 255, 0) if s[k] >= th else (255, 160, 0)
            d.rectangle([b[k][0] * W, b[k][1] * H, b[k][2] * W, b[k][3] * H], outline=col, width=2); d.text((b[k][0] * W + 2, b[k][1] * H + 1), f"{s[k]:.2f}", fill=col)
    d.rectangle([0, 0, 190, 14], fill=(0, 0, 0)); d.text((2, 1), f"{tl.filename[i][-8:-4]} GT {y[i]} pred {pr[i]}", fill=(255, 255, 0))
    im.thumbnail((S - 4, S - 4)); sheet.paste(im, ((j % 4) * S, (j // 4) * S))
sheet.save(out, quality=88)
