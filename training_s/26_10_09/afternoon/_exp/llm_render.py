"""LLM 검토용: 후보 박스에 번호를 붙여 그린다(초록 = 현재 규칙으로 센 것, 주황 = 탈락 후보). 가장자리에 원본 좌표 눈금(100px)."""
import sys, pickle, json, numpy as np, pandas as pd
sys.path.insert(0, "_exp")
from lib import _iou_inter
from PIL import Image, ImageDraw, ImageFont
p = "problem02"; tl = pd.read_csv(f"{p}/data/train_labels.csv"); V = pickle.load(open(f"_exp/{p}_views.pkl", "rb"))
ALL = ["full640", "flip640", "full960", "flip960", "full1280", "flip1280", "tile960", "full1600", "flip1600", "tile3x3", "tile2x2f"]; MAIN = {"full960": 0, "full1280": 1, "tile960": 2}
def cands(f):
    B, S, K = [], [], []
    for k in ALL:
        b, s, c = V[("train_images", f)][k]; m = c == 0; B.append(b[m]); S.append(s[m]); K += [k] * int(m.sum())
    B = np.concatenate(B); S = np.concatenate(S); reps, mem = [], []
    for i in np.argsort(-S):
        if reps:
            io, ct = _iou_inter(B[i], B[reps]); j = np.where((io >= 0.5) | (ct >= 0.8))[0]
            if len(j): mem[j[0]].append(i); continue
        reps.append(i); mem.append([i])
    out = []
    for r, mm in zip(reps, mem):
        best = {}
        for i in mm:
            if K[i] in MAIN: best[K[i]] = max(best.get(K[i], 0), S[i])
        out.append(dict(box=[float(v) for v in B[r]], conf=float(S[r]), nviews=len({K[i] for i in mm}), counted=bool(len(best) >= 2 and sum(best.values()) / 3 >= 0.15)))
    return [c for c in out if c["conf"] >= 0.05]
if __name__ == "__main__":
    allc = {}
    try: f = ImageFont.truetype("arial.ttf", 15); fs = ImageFont.truetype("arial.ttf", 11)
    except Exception: f = fs = None
    for fn, y in zip(tl.filename, tl.person_count):
        C = cands(fn); allc[fn] = C; n_cnt = sum(c["counted"] for c in C)
        if len(sys.argv) > 1 and fn[-8:-4] not in sys.argv[1:]: continue
        im = Image.open(f"{p}/data/train_images/{fn}").convert("RGB"); W, H = im.size; z = 1400 / max(W, H); im = im.resize((int(W * z), int(H * z)), Image.LANCZOS)
        cv = Image.new("RGB", (im.size[0] + 34, im.size[1] + 22), (0, 0, 0)); cv.paste(im, (34, 22)); d = ImageDraw.Draw(cv)
        for x in range(0, W, 50): d.line([34 + x * z, 16 if x % 100 == 0 else 19, 34 + x * z, 22], fill=(255, 255, 255)); (x % 100 == 0) and d.text((34 + x * z - 8, 2), str(x), fill=(255, 255, 255), font=fs)
        for yy in range(0, H, 50): d.line([28 if yy % 100 == 0 else 31, 22 + yy * z, 34, 22 + yy * z], fill=(255, 255, 255)); (yy % 100 == 0) and d.text((2, 22 + yy * z - 6), str(yy), fill=(255, 255, 255), font=fs)
        for i, c in enumerate(C):
            b = c["box"]; col = (0, 255, 0) if c["counted"] else (255, 160, 0); bx = [34 + b[0] * W * z, 22 + b[1] * H * z, 34 + b[2] * W * z, 22 + b[3] * H * z]
            d.rectangle(bx, outline=col, width=2); t = str(i); tw = 9 * len(t) + 4; d.rectangle([bx[0], bx[1], bx[0] + tw, bx[1] + 16], fill=col); d.text((bx[0] + 2, bx[1]), t, fill=(0, 0, 0), font=f)
        cv.save(f"_exp/llm/{fn[-8:-4]}.jpg", quality=90); print(fn[-8:-4], "GT", y, "counted", n_cnt, "candidates", len(C), "size", (W, H))
    json.dump(allc, open("_exp/llm/cands.json", "w"))
