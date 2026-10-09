"""yolov8n 여러 개(또는 같은 모델의 TTA)를 박스 단위로 융합(WBF)해 exact-count 를 평가한다.
사용: python p1_ens.py <split> <weights1> <weights2> ...    (split: val)
융합 점수 = 클러스터 내 박스 conf 합 / 모델 수. 임계값 이상인 클러스터 수가 그 클래스의 수량."""
import glob, sys, pickle, os, numpy as np
from ultralytics import YOLO

NAMES = ["military_tank", "military_truck", "military_vehicle", "civilian_vehicle"]

def gt_counts(i):
    c = [0] * 4
    for l in open(i.replace("images", "labels").rsplit(".", 1)[0] + ".txt").read().splitlines():
        if l.strip(): c[int(l.split()[0])] += 1
    return tuple(c)

def predict_raw(weights, paths, imgsz=640, flip=False, conf=0.05, iou=0.6, device=0):
    """한 모델의 원시 예측(정규화 xyxy, conf, cls) 목록. flip=True 면 좌우반전 이미지를 추가 추론해 박스를 되돌린다."""
    from PIL import Image
    m = YOLO(weights); out = []
    for i in range(0, len(paths), 32):
        chunk = paths[i:i + 32]
        res = m.predict(chunk, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
        if flip:
            flipped = [np.asarray(Image.open(p).convert("RGB"))[:, ::-1, ::-1].copy() for p in chunk]  # BGR 배열
            resf = m.predict(flipped, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
        for j, r in enumerate(res):
            H, W = r.orig_shape
            b = r.boxes.xyxyn.cpu().numpy(); s = r.boxes.conf.cpu().numpy(); c = r.boxes.cls.cpu().numpy().astype(int)
            parts = [(b, s, c)]
            if flip:
                rf = resf[j]; bf = rf.boxes.xyxyn.cpu().numpy().copy(); bf[:, [0, 2]] = 1 - bf[:, [2, 0]]
                parts.append((bf, rf.boxes.conf.cpu().numpy(), rf.boxes.cls.cpu().numpy().astype(int)))
            out.append(parts)
    return out  # 이미지별 [(boxes, scores, classes), ...] (전체 '관측 집합' 수 = 1 또는 2)

def iou_xyxy(a, b):
    iw = np.maximum(0, np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]))
    ih = np.maximum(0, np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1])); i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - i + 1e-9)

def wbf(sets, n_sets, thr=0.55, agnostic=False):
    """sets: [(boxes, scores, classes)] 관측 집합들. 반환: [(box, fused_score, cls)] (클러스터별)."""
    B = np.concatenate([s[0] for s in sets]) if sets else np.zeros((0, 4)); S = np.concatenate([s[1] for s in sets]) if sets else np.zeros(0)
    C = np.concatenate([s[2] for s in sets]) if sets else np.zeros(0, int)
    order = np.argsort(-S); clusters = []   # 각 클러스터: dict(members idx)
    fused = np.zeros((0, 4))
    for k in order:
        placed = False
        if len(clusters):
            ious = iou_xyxy(B[k], fused)
            cand = [(ious[j], j) for j in range(len(clusters)) if ious[j] >= thr and (agnostic or clusters[j]["cls_main"] == C[k])]
            if cand:
                j = max(cand)[1]; cl = clusters[j]; cl["idx"].append(k)
                w = S[cl["idx"]]; fused[j] = (B[cl["idx"]] * w[:, None]).sum(0) / w.sum(); placed = True
        if not placed:
            clusters.append({"idx": [k], "cls_main": C[k]}); fused = np.vstack([fused, B[k]])
    out = []
    for j, cl in enumerate(clusters):
        idx = cl["idx"]; score = S[idx].sum() / n_sets
        # 클래스는 클러스터 안 conf 합이 가장 큰 클래스
        cls_sum = np.bincount(C[idx], weights=S[idx], minlength=4); out.append((fused[j], float(score), int(cls_sum.argmax()), cls_sum / n_sets))
    return out

def counts_from(clusters, thr):
    c = [0] * 4
    for _, s, k, _ in clusters:
        if s >= thr: c[k] += 1
    return tuple(c)

if __name__ == "__main__":
    split = sys.argv[1]; weights = sys.argv[2:]
    paths = sorted(glob.glob(f"data/{split}/images/*")); G = [gt_counts(p) for p in paths]
    raws = [predict_raw(w, paths) for w in weights]
    T = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)
    for agn in (False, True):
        clusters = [wbf([raws[m][i][0] for m in range(len(weights))], len(weights), agnostic=agn) for i in range(len(paths))]
        a = {t: np.mean([counts_from(cl, t) == g for cl, g in zip(clusters, G)]) for t in T}
        bt = max(a, key=a.get)
        print(f"ens{len(weights)} agn{int(agn)} best {a[bt]:.4f}@{bt} | " + " ".join(f"{t}:{a[t]:.3f}" for t in T))
