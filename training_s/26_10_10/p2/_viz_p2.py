import os, numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
AO, TL = "data/aerial_objects", "data/transfer_learning"; OUT = "_viz"; os.makedirs(OUT, exist_ok=True)
D = pd.read_csv("_p2_coco_pass.csv")
def font(s):
    for p in ("C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf"):
        if os.path.exists(p): return ImageFont.truetype(p, s)
    return ImageFont.load_default()
def sheet(items, cols, cell, path, title):
    rows = (len(items) + cols - 1) // cols; S = Image.new("RGB", (cols * cell[0], rows * cell[1] + 34), (252, 252, 251)); dr = ImageDraw.Draw(S); dr.text((8, 6), title, fill=(11, 11, 11), font=font(16))
    for i, (img, cap) in enumerate(items):
        r = min((cell[0] - 6) / img.size[0], (cell[1] - 24) / img.size[1]); t = img.resize((max(1, int(img.size[0] * r)), max(1, int(img.size[1] * r))), Image.LANCZOS)
        x, y = (i % cols) * cell[0], 34 + (i // cols) * cell[1]; S.paste(t, (x + 3 + (cell[0] - 6 - t.size[0]) // 2, y + 20)); dr.text((x + 4, y + 2), cap, fill=(11, 11, 11), font=font(12))
    S.save(path, quality=88)
none = D[D.kind.str.contains("없음")].sort_values("f")
sheet([(Image.open(os.path.join(AO, f)).convert("RGB"), f"{f[:-4]}  {mp:.1f}MP") for f, mp in zip(none.f, none.mp)], 8, (230, 190), os.path.join(OUT, "test_no_airplane_bird.jpg"), f"테스트 중 사전학습 모델이 airplane·bird 를 못 찾은 {len(none)}장 (드론 후보)")
tl = sorted(f for f in os.listdir(TL) if f.lower().endswith((".jpg", ".png")))[:40]; items = []
for f in tl:
    im = Image.open(os.path.join(TL, f)).convert("RGB"); W, H = im.size; dr = ImageDraw.Draw(im)
    for l in open(os.path.join(TL, os.path.splitext(f)[0] + ".txt")).read().splitlines():
        c, x, y, w, h = map(float, l.split()); dr.rectangle([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H], outline=(235, 104, 52), width=max(3, W // 150))
    items.append((im, f"{f}  {W}x{H}"))
sheet(items, 8, (230, 190), os.path.join(OUT, "transfer_drone_boxes.jpg"), "전이학습 이미지 40장 (주황 = 드론 라벨 박스)")
for kind, n in (("airplane만", 24), ("bird만", 24)):
    sub = D[D.kind == kind].sort_values("f").iloc[:: max(1, (D.kind == kind).sum() // n)].head(n)
    sheet([(Image.open(os.path.join(AO, f)).convert("RGB"), f"{f[:-4]}  air {a} bird {b}") for f, a, b in zip(sub.f, sub.n_air, sub.n_bird)], 8, (230, 190), os.path.join(OUT, f"test_{'airplane' if kind.startswith('air') else 'bird'}.jpg"), f"테스트: 사전학습이 {kind} 로 본 이미지 표본")
print(os.listdir(OUT))
