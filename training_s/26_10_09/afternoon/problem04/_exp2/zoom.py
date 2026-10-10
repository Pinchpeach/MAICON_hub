"""확대 검토: python zoom.py 파일 x0 y0 x1 y1 [파일 ...] (좌표는 전체 이미지 기준 %). 후보 박스와 전체 기준 % 눈금을 함께 그린다."""
import sys, os, json; sys.path.insert(0, os.path.dirname(__file__))
from PIL import Image, ImageDraw, ImageEnhance
from harness import DATA, ROOT
import make_dummy as M
OUT = os.path.join(ROOT, "_exp2", "review"); C = json.load(open(os.path.join(OUT, "candidates.json"))); a = sys.argv[1:]
for i in range(0, len(a), 5):
    f, x0, y0, x1, y1 = a[i], *map(float, a[i + 1:i + 5]); img = Image.open(os.path.join(DATA, "train_images", f)).convert("RGB"); W, H = img.size
    box = (int(W * x0 / 100), int(H * y0 / 100), int(W * x1 / 100), int(H * y1 / 100)); cr = img.crop(box); sc = min(1200 / cr.size[0], 900 / cr.size[1]); cr = cr.resize((int(cr.size[0] * sc), int(cr.size[1] * sc)), Image.LANCZOS)
    if "bright" in os.environ.get("ZOOM", ""): cr = ImageEnhance.Brightness(cr).enhance(2.5)
    cw, ch = cr.size; pad = 30; full = Image.new("RGB", (cw + pad, ch + pad), (252, 252, 251)); full.paste(cr, (pad, pad)); dr = ImageDraw.Draw(full); fs = M.font(12); span = max(x1 - x0, y1 - y0); step = 1 if span <= 12 else (2 if span <= 25 else 5)
    px = lambda x: pad + (x - x0) / (x1 - x0) * cw; py = lambda y: pad + (y - y0) / (y1 - y0) * ch
    v = int(x0 // step * step)
    while v <= x1:
        if v >= x0: dr.line([px(v), pad, px(v), pad + ch], fill=(255, 255, 255), width=1); dr.text((px(v) - 6, 12), str(v), fill=(11, 11, 11), font=fs)
        v += step
    v = int(y0 // step * step)
    while v <= y1:
        if v >= y0: dr.line([pad, py(v), pad + cw, py(v)], fill=(255, 255, 255), width=1); dr.text((2, py(v) - 6), str(v), fill=(11, 11, 11), font=fs)
        v += step
    for c in C.get(f, []):
        b = [c["box"][0] * 100, c["box"][1] * 100, c["box"][2] * 100, c["box"][3] * 100]
        if b[2] < x0 or b[0] > x1 or b[3] < y0 or b[1] > y1: continue
        col = M.COL[c["group"]] if c["kept"] else ((120, 160, 215) if c["group"] == "person" else (205, 150, 120)); dr.rectangle([px(b[0]), py(b[1]), px(b[2]), py(b[3])], outline=col, width=2 if c["kept"] else 1)
        dr.text((max(px(b[0]), pad) + 3, max(py(b[1]), pad) + 2), f"{c['id']} {c['score']:.2f}", fill=col, font=M.font(14), stroke_width=2, stroke_fill=(255, 255, 255))
    name = f"zoom_{f[:-4]}_{int(x0)}_{int(y0)}.jpg"; full.save(os.path.join(OUT, name), quality=92); print(name)
