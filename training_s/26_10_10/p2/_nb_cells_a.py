CELL_B = '''import os, json, random, shutil, glob, time, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from PIL import Image
from ultralytics import YOLO

DEVICE = 0 if torch.cuda.is_available() else "cpu"     # GPU 가 없으면 CPU 로 같은 코드 실행
random.seed(0); np.random.seed(0)
TL, AO = "data/transfer_learning", "data/aerial_objects"; WORK = "work_p2"; os.makedirs(WORK, exist_ok=True)
MAXS = 640                                              # 학습 이미지를 파일로 쓸 때 긴 변 상한(어차피 imgsz 640)

def load(f):
    """전이학습 이미지 1장과 박스(정규화 xywh)를 읽는다. 긴 변이 640 을 넘으면 줄인다."""
    im = Image.open(os.path.join(TL, f)).convert("RGB"); w, h = im.size; s = MAXS / max(w, h)
    if s < 1: im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    x, y, bw, bh = map(float, open(os.path.join(TL, os.path.splitext(f)[0] + ".txt")).read().split()[1:5])
    return im, [(x, y, bw, bh)]

def dihedral(im, boxes, op):
    """상하 반전·90/180/270도 회전과 박스 변환(박스는 정규화 xywh). 좌우 반전은 학습이 온라인으로 하므로 제외."""
    if op == "vflip": return im.transpose(Image.FLIP_TOP_BOTTOM), [(x, 1 - y, w, h) for x, y, w, h in boxes]
    if op == "rot90": return im.transpose(Image.ROTATE_90), [(y, 1 - x, h, w) for x, y, w, h in boxes]
    if op == "rot180": return im.transpose(Image.ROTATE_180), [(1 - x, 1 - y, w, h) for x, y, w, h in boxes]
    if op == "rot270": return im.transpose(Image.ROTATE_270), [(1 - y, x, h, w) for x, y, w, h in boxes]
    return im, boxes

def crop_drone(im, box, margin=0.06):
    W, H = im.size; x, y, w, h = box
    x0 = max(0, (x - w / 2 - margin * w) * W); y0 = max(0, (y - h / 2 - margin * h) * H); x1 = min(W, (x + w / 2 + margin * w) * W); y1 = min(H, (y + h / 2 + margin * h) * H)
    return im.crop((int(x0), int(y0), max(int(x0) + 4, int(x1)), max(int(y0) + 4, int(y1))))

def synth(target, boxes, donors, rng, k):
    """대상 이미지에 다른 이미지에서 오린 드론 k 개를 겹치지 않게 붙이고 박스를 추가한다(다중 드론 합성)."""
    im = target.copy(); W, H = im.size; out = list(boxes); placed = tries = 0
    while placed < k and tries < 60:
        tries += 1; dim, db = rng.choice(donors); c = crop_drone(dim, db[0])
        if rng.random() < 0.5: c = c.transpose(Image.FLIP_LEFT_RIGHT)
        r = rng.uniform(0.15, 0.42) * min(W, H) / max(c.size); cw, ch = max(4, int(c.size[0] * r)), max(4, int(c.size[1] * r))
        if cw >= W or ch >= H: continue
        px, py = rng.randint(0, W - cw), rng.randint(0, H - ch); nb = ((px + cw / 2) / W, (py + ch / 2) / H, cw / W, ch / H); ok = True
        for x, y, w, h in out:
            iw = max(0, min(nb[0] + nb[2] / 2, x + w / 2) - max(nb[0] - nb[2] / 2, x - w / 2)); ih = max(0, min(nb[1] + nb[3] / 2, y + h / 2) - max(nb[1] - nb[3] / 2, y - h / 2))
            if iw * ih > 0.02 * nb[2] * nb[3]: ok = False; break
        if not ok: continue
        im.paste(c.resize((cw, ch), Image.LANCZOS), (px, py)); out.append(nb); placed += 1
    return im, out

def write(d, name, im, boxes):
    os.makedirs(f"{d}/images", exist_ok=True); os.makedirs(f"{d}/labels", exist_ok=True); im.save(f"{d}/images/{name}.jpg", quality=92)
    open(f"{d}/labels/{name}.txt", "w").write("\\n".join("0 %.6f %.6f %.6f %.6f" % b for b in boxes) + "\\n")

def build_train_set(seed=0):
    """전이학습 58장 전체 + 이미지당 회전·반전 2개 + 이미지당 다중 합성 1개. 반환: data.yaml 경로, 학습 이미지 수"""
    rng = random.Random(seed); base = f"{WORK}/dataset"; shutil.rmtree(base, ignore_errors=True)
    files = sorted(f for f in os.listdir(TL) if f.lower().endswith((".jpg", ".jpeg", ".png"))); T = {f: load(f) for f in files}; donors = [(T[f][0], T[f][1]) for f in files]
    for f in files:
        im, bx = T[f]; stem = os.path.splitext(f)[0]; write(f"{base}/train", stem, im, bx); write(f"{base}/val", stem, im, bx)      # val 은 원본(모니터링용, 독립 검증 아님)
        for op in rng.sample(["vflip", "rot90", "rot180", "rot270"], 2): i2, b2 = dihedral(im, bx, op); write(f"{base}/train", f"{stem}_{op}", i2, b2)
        i2, b2 = synth(im, bx, donors, rng, rng.choice([1, 2, 3])); write(f"{base}/train", stem + "_syn", i2, b2)
    yml = f"{base}/data.yaml"; open(yml, "w", encoding="utf-8").write("path: %s\\ntrain: train/images\\nval: val/images\\nnames:\\n  0: drone\\n" % json.dumps(os.path.abspath(base)))
    return os.path.abspath(yml), len(glob.glob(f"{base}/train/images/*"))
print("device:", DEVICE, "| torch", torch.__version__, "| 전이학습 이미지", len([f for f in os.listdir(TL) if f.lower().endswith((".jpg", ".jpeg", ".png"))]), "| 테스트", len(os.listdir(AO)))
'''

CELL_C = '''# ---- 드론 전이학습: yolov8s, 학습률 0.0002, 20에폭 (그룹 교차검증에서 고른 설정) ----
yml, n_train = build_train_set()
kw = dict(epochs=20, imgsz=640, batch=8, optimizer="AdamW", lr0=0.0002, patience=0, seed=0, deterministic=True, workers=0, plots=False, verbose=False, exist_ok=True, cache="ram",
          project=os.path.abspath(f"{WORK}/runs"), name="drone_s")
if DEVICE == "cpu": kw["time"] = 0.4                     # CPU 에서는 시간 상한(0.4시간)을 둔다 -> 에폭 수가 줄 수 있음
t0 = time.time(); drone_model = YOLO("yolov8s.pt"); res = drone_model.train(data=yml, device=DEVICE, **kw); t_train = time.time() - t0
DRONE_W = os.path.join(str(drone_model.trainer.save_dir), "weights", "last.pt")
print(f"학습 이미지 {n_train}장, 학습 {t_train:.0f}s ({t_train / kw['epochs']:.1f}s/에폭, {DEVICE}) | 가중치: {DRONE_W}")
'''

CELL_D = '''# ---- 테스트 300장 추론: 드론 모델(전이학습)과 기본 yolov8s(비행기·새) ----
test_files = sorted(f for f in os.listdir(AO) if f.lower().endswith(".jpg")); assert len(test_files) == 300
def predict_all(weights, conf=0.05, iou=0.6, bs=10):
    m = YOLO(weights); out = {}
    for i in range(0, len(test_files), bs):
        for f, r in zip(test_files[i:i + bs], m.predict([os.path.join(AO, x) for x in test_files[i:i + bs]], imgsz=640, conf=conf, iou=iou, device=DEVICE, verbose=False)):
            out[f] = (r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(), r.boxes.xyxyn.cpu().numpy())
    return out
t0 = time.time(); P_drone = predict_all(DRONE_W); P_base = predict_all("yolov8s.pt"); t_pred = time.time() - t0
print(f"추론 {t_pred:.0f}s (모델 2개 x 300장, {DEVICE})")
'''
