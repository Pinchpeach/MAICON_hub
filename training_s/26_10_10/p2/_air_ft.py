"""비행기 전용 전이학습(yolov8s 유지). 기본 모델이 한 대로 확신하는 테스트 이미지(답지 미사용)의 가짜 박스로, 작게 줄여 일부를 가린 비행기를 여러 대 붙인 합성 이미지를 만들어 학습한다.
사용 범위: 테스트 이미지의 입력 + 기본 모델 예측만 사용(정답 미사용). 임계값은 떼어 둔 기증 이미지의 합성 검증으로 정한다(대리 지표)."""
import os, json, random, shutil, glob, time, pickle, sys, numpy as np, pandas as pd, torch
from PIL import Image
from ultralytics import YOLO
AIR, BIRD = 4, 14; AO, TL = "data/aerial_objects", "data/transfer_learning"; WORK = "work_air"; DEV = 0 if torch.cuda.is_available() else "cpu"; MAXS = 640
S = pd.read_csv("submission.csv").set_index("filename"); PB = pickle.load(open("_final_preds.pkl", "rb"))["P_base"]; files = list(S.index)
def top(f, cid):
    c, s, b = PB[f]; m = c == cid; return (float(s[m].max()), b[m][np.argmax(s[m])]) if m.any() else (0.0, None)
PDr = pickle.load(open("_v3_drone_pred.pkl", "rb")); sd = lambda f: float(PDr[f][0].max()) if len(PDr[f][0]) else 0.0                 # 드론 모델 점수(드론이 기증 이미지에 섞이지 않게)
donors = [f for f in files if S.loc[f, "airplane"] > 0 and top(f, AIR)[0] >= 0.8 and ((PB[f][0] == AIR) & (PB[f][1] >= 0.1)).sum() == 1 and sd(f) < 0.1]      # 한 대로 확신
neg_bird = [f for f in files if top(f, BIRD)[0] >= 0.7 and top(f, AIR)[0] < 0.3]
neg_drone = sorted(f for f in os.listdir(TL) if f.lower().endswith((".jpg", ".jpeg", ".png")))
rng = random.Random(0); sh = donors[:]; rng.shuffle(sh); val_d = sorted(sh[:14]); tr_d = sorted(sh[14:])
print("기증 이미지(한 대 확신) %d장 -> 학습 %d / 검증 %d | 부정: 새 %d, 드론 학습 이미지 %d" % (len(donors), len(tr_d), len(val_d), len(neg_bird), len(neg_drone)))
def load(path):
    im = Image.open(path).convert("RGB"); w, h = im.size; s = MAXS / max(w, h)
    return im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS) if s < 1 else im
IM = {f: load(os.path.join(AO, f)) for f in donors}; BX = {f: top(f, AIR)[1] for f in donors}       # 정규화 xyxy
def crop(f):
    im = IM[f]; W, H = im.size; x0, y0, x1, y1 = BX[f]; return im.crop((int(x0 * W), int(y0 * H), max(int(x0 * W) + 4, int(x1 * W)), max(int(y0 * H) + 4, int(y1 * H))))
def synth(f, pool, rng, k):
    """대상 이미지에 작은 비행기 k 대를 붙인 뒤, 원래 큰 비행기의 가운데 띠(동체)를 다시 덮어 작은 비행기가 뒤에 가려진 것처럼 만든다. 박스는 보이는 부분만(픽셀 xyxy)."""
    src = IM[f]; im = src.copy(); W, H = im.size; mx0, my0, mx1, my1 = [v * s for v, s in zip(BX[f], (W, H, W, H))]; mh = my1 - my0
    band = (int(mx0), int(my0 + 0.3 * mh), int(mx1), int(my0 + 0.7 * mh)); small = []; tries = 0
    while len(small) < k and tries < 80:
        tries += 1; c = crop(rng.choice(pool))
        if rng.random() < 0.5: c = c.transpose(Image.FLIP_LEFT_RIGHT)
        cw = int(rng.uniform(0.06, 0.28) * W); ch = max(4, int(cw * c.size[1] / c.size[0]))
        if cw < 8 or cw >= W or ch >= H: continue
        px, py = rng.randint(-cw // 3, W - cw + cw // 3), rng.randint(0, H - ch); b = [max(0, px), py, min(W, px + cw), py + ch]      # 좌우 가장자리에서 잘릴 수 있음
        if (b[2] - b[0]) < 0.5 * cw: continue
        # 가운데 띠에 가려지는 부분을 뺀 보이는 영역(위쪽 조각 또는 아래쪽 조각 중 큰 쪽)
        if b[0] < band[2] and b[2] > band[0] and b[1] < band[3] and b[3] > band[1]:
            if not (b[0] >= band[0] and b[2] <= band[2]): continue                       # 띠에 걸치려면 가로로는 띠 안에 완전히 들어가야 박스가 단순해진다
            up, dn = max(0, band[1] - b[1]), max(0, b[3] - band[3])
            if max(up, dn) < 0.35 * ch: continue
            b = [b[0], b[1], b[2], band[1]] if up >= dn else [b[0], band[3], b[2], b[3]]
        ok = True
        for q in small:
            iw = max(0, min(b[2], q[2]) - max(b[0], q[0])); ih = max(0, min(b[3], q[3]) - max(b[1], q[1]))
            if iw * ih > 0.05 * min((b[2] - b[0]) * (b[3] - b[1]), (q[2] - q[0]) * (q[3] - q[1])): ok = False; break
        if not ok: continue
        im.paste(c.resize((cw, ch), Image.LANCZOS), (px, py)); small.append(b)
    im.paste(src.crop(band), (band[0], band[1]))                                         # 큰 비행기의 동체 띠를 다시 덮는다
    return im, [[mx0, my0, mx1, my1]] + small
def write(d, name, im, boxes):
    os.makedirs(d + "/images", exist_ok=True); os.makedirs(d + "/labels", exist_ok=True); W, H = im.size; im.save("%s/images/%s.jpg" % (d, name), quality=92)
    open("%s/labels/%s.txt" % (d, name), "w").write("".join("0 %.6f %.6f %.6f %.6f\n" % ((b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H, (b[2] - b[0]) / W, (b[3] - b[1]) / H) for b in boxes))
base = WORK + "/dataset"; shutil.rmtree(base, ignore_errors=True); VAL = {}
for f in tr_d:
    st = os.path.splitext(f)[0]; write(base + "/train", st, *synth(f, tr_d, rng, 0))
    for j in range(4): write(base + "/train", "%s_s%d" % (st, j), *synth(f, [g for g in tr_d if g != f], rng, rng.choice([1, 2, 3, 4, 5, 6])))
for f in val_d:
    st = os.path.splitext(f)[0]
    for j, k in enumerate((0, 1, 2, 3, 5)):
        im, bx = synth(f, [g for g in val_d if g != f], rng, k); write(base + "/val", "%s_v%d" % (st, j), im, bx); VAL["%s_v%d" % (st, j)] = len(bx)
for f in neg_bird: write(base + "/train", "negb_" + os.path.splitext(f)[0], load(os.path.join(AO, f)), [])
for f in neg_drone: write(base + "/train", "negd_" + os.path.splitext(f)[0], load(os.path.join(TL, f)), [])
yml = os.path.abspath(base + "/data.yaml"); open(yml, "w", encoding="utf-8").write("path: %s\ntrain: train/images\nval: val/images\nnames:\n  0: airplane\n" % json.dumps(os.path.abspath(base)))
print("학습 이미지 %d장, 검증(합성) %d장" % (len(glob.glob(base + "/train/images/*")), len(VAL)))
if len(sys.argv) > 1 and sys.argv[1] == "build": sys.exit()
t0 = time.time(); m = YOLO("yolov8s.pt")
m.train(data=yml, device=DEV, epochs=20, imgsz=640, batch=8, optimizer="AdamW", lr0=0.0002, patience=0, seed=0, deterministic=True, workers=0, plots=False, verbose=False, exist_ok=True, cache="ram", project=os.path.abspath(WORK + "/runs"), name="air_s")
print("학습 %.0fs (%s)" % (time.time() - t0, DEV)); Wt = os.path.join(str(m.trainer.save_dir), "weights", "last.pt"); m = YOLO(Wt)
def pred(paths):
    out = []
    for i in range(0, len(paths), 10):
        for r in m.predict(paths[i:i + 10], imgsz=640, conf=0.05, iou=0.6, device=DEV, verbose=False): out.append((r.boxes.conf.cpu().numpy(), r.boxes.xyxyn.cpu().numpy()))
    return out
vn = sorted(VAL); PV = dict(zip(vn, pred(["%s/val/images/%s.jpg" % (base, n) for n in vn]))); af = [f for f in files if S.loc[f, "airplane"] > 0]; PA = dict(zip(af, pred([os.path.join(AO, f) for f in af])))
pickle.dump(dict(VAL=VAL, PV=PV, PA=PA, donors=donors, val_d=val_d), open("_air_ft.pkl", "wb")); print("saved", Wt)
