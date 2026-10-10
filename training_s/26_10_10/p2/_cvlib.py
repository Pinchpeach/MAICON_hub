"""그룹 교차검증용 데이터 생성: 오프라인 회전·반전 증강, 다중 드론 합성. 모든 변환은 겹의 학습 이미지에만 적용한다."""
import os, json, random, shutil
import numpy as np
from PIL import Image
SRC = "data/transfer_learning"; MAXS = 640                      # 파일로 쓸 때 긴 변을 640 으로 줄임(어차피 imgsz 640)

def load(f):
    im = Image.open(os.path.join(SRC, f)).convert("RGB"); w, h = im.size; s = MAXS / max(w, h)
    if s < 1: im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    x, y, bw, bh = map(float, open(os.path.join(SRC, os.path.splitext(f)[0] + ".txt")).read().split()[1:5])
    return im, [(x, y, bw, bh)]

def dihedral(im, boxes, op):
    """op: vflip, rot90(반시계), rot180, rot270(반시계 270 = 시계 90). 박스는 정규화 xywh."""
    if op == "vflip": return im.transpose(Image.FLIP_TOP_BOTTOM), [(x, 1 - y, w, h) for x, y, w, h in boxes]
    if op == "rot90": return im.transpose(Image.ROTATE_90), [(y, 1 - x, h, w) for x, y, w, h in boxes]
    if op == "rot180": return im.transpose(Image.ROTATE_180), [(1 - x, 1 - y, w, h) for x, y, w, h in boxes]
    if op == "rot270": return im.transpose(Image.ROTATE_270), [(1 - y, x, h, w) for x, y, w, h in boxes]
    return im, boxes

def crop_drone(im, box, margin=0.06):
    W, H = im.size; x, y, w, h = box; x0 = max(0, (x - w / 2 - margin * w) * W); y0 = max(0, (y - h / 2 - margin * h) * H); x1 = min(W, (x + w / 2 + margin * w) * W); y1 = min(H, (y + h / 2 + margin * h) * H)
    return im.crop((int(x0), int(y0), max(int(x0) + 4, int(x1)), max(int(y0) + 4, int(y1))))

def synth(target, boxes, donors, rng, k):
    """target 이미지에 donors(이미지, 박스) 에서 오린 드론 k 개를 겹치지 않게 붙인다. 반환: 새 이미지, 새 박스 목록"""
    im = target.copy(); W, H = im.size; out = list(boxes); placed = 0; tries = 0
    while placed < k and tries < 60:
        tries += 1; dim, db = rng.choice(donors); c = crop_drone(dim, db[0])
        if rng.random() < 0.5: c = c.transpose(Image.FLIP_LEFT_RIGHT)
        r = rng.uniform(0.15, 0.42) * min(W, H) / max(c.size); cw, ch = max(4, int(c.size[0] * r)), max(4, int(c.size[1] * r))
        if cw >= W or ch >= H: continue
        px, py = rng.randint(0, W - cw), rng.randint(0, H - ch); nb = ((px + cw / 2) / W, (py + ch / 2) / H, cw / W, ch / H)
        ok = True
        for x, y, w, h in out:
            iw = max(0, min(nb[0] + nb[2] / 2, x + w / 2) - max(nb[0] - nb[2] / 2, x - w / 2)); ih = max(0, min(nb[1] + nb[3] / 2, y + h / 2) - max(nb[1] - nb[3] / 2, y - h / 2))
            if iw * ih > 0.02 * nb[2] * nb[3]: ok = False; break
        if not ok: continue
        im.paste(c.resize((cw, ch), Image.LANCZOS), (px, py)); out.append(nb); placed += 1
    return im, out

def write(d, name, im, boxes):
    os.makedirs(f"{d}/images", exist_ok=True); os.makedirs(f"{d}/labels", exist_ok=True); im.save(f"{d}/images/{name}.jpg", quality=92)
    open(f"{d}/labels/{name}.txt", "w").write("\n".join("0 %.6f %.6f %.6f %.6f" % b for b in boxes) + "\n")

def build_fold(cfg, k, folds, root="_cv", seed=0):
    """cfg: set 의 원소 'dihedral', 'synth'. 반환: (yaml 경로, 학습 이미지 수, 합성 검증 디렉터리)"""
    rng = random.Random(seed * 100 + k); val = folds[k]; tr = [f for i, fo in enumerate(folds) if i != k for f in fo]; name = "+".join(sorted(cfg)) or "base"
    base = f"{root}/{name}/f{k}"; shutil.rmtree(base, ignore_errors=True)
    T = {f: load(f) for f in tr}
    for f in tr:
        im, bx = T[f]; stem = os.path.splitext(f)[0]; write(f"{base}/train", stem, im, bx)
        if "dihedral" in cfg:
            for op in rng.sample(["vflip", "rot90", "rot180", "rot270"], 2): i2, b2 = dihedral(im, bx, op); write(f"{base}/train", f"{stem}_{op}", i2, b2)
    if "synth" in cfg:
        donors = [(T[f][0], T[f][1]) for f in tr]
        for f in tr:
            im, bx = T[f]; i2, b2 = synth(im, bx, donors, rng, rng.choice([1, 2, 3])); write(f"{base}/train", os.path.splitext(f)[0] + "_syn", i2, b2)
    for f in val: im, bx = load(f); write(f"{base}/val", os.path.splitext(f)[0], im, bx)
    yml = f"{base}/data.yaml"; open(yml, "w", encoding="utf-8").write("path: %s\ntrain: train/images\nval: val/images\nnames:\n  0: drone\n" % json.dumps(os.path.abspath(base)))
    return os.path.abspath(yml)

def build_synth_val(k, folds, root="_cv", seed=1, n_per=2):
    """검증 이미지끼리만 합성한 다중 드론 이미지(개수 정확도 대리 지표). 모든 구성이 공유한다."""
    d = f"{root}/synthval/f{k}"
    if os.path.exists(d): return d
    rng = random.Random(seed * 100 + k); V = {f: load(f) for f in folds[k]}; donors = [(im, bx) for im, bx in V.values()]
    for f, (im, bx) in V.items():
        for j in range(n_per): i2, b2 = synth(im, bx, donors, rng, rng.choice([1, 2, 3])); write(d, f"{os.path.splitext(f)[0]}_s{j}", i2, b2)
    return d
