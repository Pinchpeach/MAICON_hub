"""전이학습 58장을 학습 47 / 검증 11 로 나눠 YOLO 형식 폴더와 yaml 을 만든다(원본 data 는 건드리지 않음)."""
import os, shutil, random, json
SRC = "data/transfer_learning"; DST = os.path.abspath("_tl_data").replace("\\", "/"); random.seed(0)
imgs = sorted(f for f in os.listdir(SRC) if f.lower().endswith((".jpg", ".jpeg", ".png")))
val = set(random.sample(imgs, 11)); shutil.rmtree(DST, ignore_errors=True)
for sp in ("train", "val"):
    for k in ("images", "labels"): os.makedirs(f"{DST}/{k}/{sp}")
for f in imgs:
    sp = "val" if f in val else "train"; stem = os.path.splitext(f)[0]
    shutil.copy(os.path.join(SRC, f), f"{DST}/images/{sp}/{f}"); shutil.copy(os.path.join(SRC, stem + ".txt"), f"{DST}/labels/{sp}/{stem}.txt")
open("_tl_data.yaml", "w", encoding="utf-8").write(f"path: {json.dumps(DST)}\ntrain: images/train\nval: images/val\nnames:\n  0: drone\n")
# 전체 58장 학습용(최종 후보): 검증도 같은 폴더를 보지만 모니터링용
os.makedirs(f"{DST}/images/all", exist_ok=True); os.makedirs(f"{DST}/labels/all", exist_ok=True)
for f in imgs:
    stem = os.path.splitext(f)[0]; shutil.copy(os.path.join(SRC, f), f"{DST}/images/all/{f}"); shutil.copy(os.path.join(SRC, stem + ".txt"), f"{DST}/labels/all/{stem}.txt")
open("_tl_all.yaml", "w", encoding="utf-8").write(f"path: {json.dumps(DST)}\ntrain: images/all\nval: images/val\nnames:\n  0: drone\n")
print("학습 %d / 검증 %d / 전체 %d" % (len(imgs) - len(val), len(val), len(imgs)), "| 검증 이미지:", sorted(val)[:11])
