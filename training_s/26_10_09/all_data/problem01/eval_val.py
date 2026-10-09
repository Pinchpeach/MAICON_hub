import sys, glob, os, itertools, numpy as np
from collections import Counter
from ultralytics import YOLO

weights, imgsz, dev = sys.argv[1], int(sys.argv[2]), sys.argv[3]
model = YOLO(weights)
imgs = sorted(glob.glob("data/val/images/*"))

def gt_counts(img):
    lab = img.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
    c = [0] * 4
    for l in open(lab).read().splitlines():
        if l.strip(): c[int(l.split()[0])] += 1
    return tuple(c)

gt = [gt_counts(i) for i in imgs]
# 낮은 conf 로 한 번만 추론해 박스를 저장한 뒤, conf 임계값을 후처리로 바꿔 가며 평가
res = model.predict(imgs, imgsz=imgsz, conf=0.05, iou=0.6, device=dev, verbose=False, augment=False)
boxes = [(r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()) for r in res]

def counts(c, thr):
    out = [0] * 4
    for k, s in zip(*c):
        t = thr[k] if isinstance(thr, (list, tuple)) else thr
        if s >= t: out[k] += 1
    return tuple(out)

def acc(thr): return np.mean([counts(b, thr) == g for b, g in zip(boxes, gt)])

print("전부 0 예측 정확도: %.4f" % np.mean([g == (0, 0, 0, 0) for g in gt]))
for t in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6): print("conf %.2f  exact acc %.4f" % (t, acc(t)))
best = max((0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6), key=acc)
print("best global conf", best, "acc %.4f" % acc(best))
# 클래스별 임계값 좌표 하강 (검증 160장에 맞춘 값이라 낙관적일 수 있음)
thr = [best] * 4
for _ in range(2):
    for k in range(4):
        thr[k] = max((0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7), key=lambda v: acc(thr[:k] + [v] + thr[k + 1:]))
print("per-class thr", thr, "acc %.4f (검증에 맞춘 값, 낙관적)" % acc(thr))
# 오답 분석: 어떤 클래스에서 틀리는지
wrong = [(g, counts(b, best)) for b, g in zip(boxes, gt) if counts(b, best) != g]
over = sum(1 for g, p in wrong if sum(p) > sum(g)); under = sum(1 for g, p in wrong if sum(p) < sum(g))
print("오답 %d건: 과다예측 %d, 과소예측 %d, 총수는 같고 클래스만 다름 %d" % (len(wrong), over, under, len(wrong) - over - under))
cls_err = Counter()
for g, p in wrong:
    for k in range(4):
        if g[k] != p[k]: cls_err[k] += 1
print("클래스별 수량 불일치 이미지 수:", dict(cls_err))
