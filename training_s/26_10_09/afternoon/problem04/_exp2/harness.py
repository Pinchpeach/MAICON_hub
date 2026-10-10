"""문제 4 실험 도구 (이 세션 전용, 이름 _exp2). 메인 노트북의 merge 규칙은 그대로 쓰고, 추론만 '후처리 전 출력'에서 직접 만든다."""
import os, pickle, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torchvision
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DATA = os.path.join(ROOT, "data")
labels = pd.read_csv(os.path.join(DATA, "train_labels.csv")); sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
PERSON, VEH = 0, [2, 5, 7]                       # COCO: person=0, car=2, bus=5, truck=7
_model = None


def net():
    global _model
    if _model is None: _model = YOLO(os.path.join(ROOT, "yolov8n.pt")); _model.model.eval()
    return _model


@torch.no_grad()
def raw_pred(arr_bgr, imgsz):
    """후처리(NMS) 전 출력. 반환: 정규화 xyxy (N,4), 클래스 확률 (N,80)"""
    H, W = arr_bgr.shape[:2]; lb = LetterBox(new_shape=(imgsz, imgsz), auto=True, stride=32)(image=arr_bgr)
    x = torch.from_numpy(np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1))).float().unsqueeze(0) / 255.0
    out = net().model(x); out = out[0] if isinstance(out, (list, tuple)) else out          # (1, 84, N)
    p = out[0].T; box = ops.xywh2xyxy(p[:, :4]); cls = p[:, 4:]
    box = ops.scale_boxes(lb.shape[:2], box.clone(), (H, W)); box = box / torch.tensor([W, H, W, H])
    return box.numpy(), cls.numpy()


def group_views(path, sizes=(640, 960, 1280), flips=(False, True)):
    """이미지 1장 -> {(imgsz, flip): (box, cls)}"""
    arr = np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1]; out = {}
    for s in sizes:
        for f in flips:
            b, c = raw_pred(arr[:, ::-1].copy() if f else arr, s)
            if f: b = b.copy(); b[:, [0, 2]] = 1 - b[:, [2, 0]]
            out[(s, f)] = (b, c[:, [PERSON] + VEH].copy())                                  # 필요한 4개 클래스만 저장
    return out


def to_dets(view, group, combine="max", pre_conf=0.03, nms_iou=0.7):
    """한 관점의 원시 출력 -> 그룹 점수로 NMS 한 (box, score). combine: max(기존과 같음) / sum(합, 1 로 자름) / nor(1-곱(1-p))"""
    b, c = view
    if group == "person": s = c[:, 0]
    else:
        v = c[:, 1:4]; s = v.max(1) if combine == "max" else (np.minimum(v.sum(1), 1.0) if combine == "sum" else 1 - np.prod(1 - v, axis=1))
    m = s >= pre_conf; b, s = b[m], s[m]
    if not len(s): return np.zeros((0, 4)), np.zeros(0)
    keep = torchvision.ops.nms(torch.from_numpy(b).float(), torch.from_numpy(s).float(), nms_iou).numpy()[:300]
    return b[keep], s[keep]


def _iou_inter(a, B):
    iw = np.maximum(0, np.minimum(a[2], B[:, 2]) - np.maximum(a[0], B[:, 0])); ih = np.maximum(0, np.minimum(a[3], B[:, 3]) - np.maximum(a[1], B[:, 1]))
    inter = iw * ih; aa = (a[2] - a[0]) * (a[3] - a[1]); ab = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (aa + ab - inter + 1e-12), inter / (np.minimum(aa, ab) + 1e-12)


def merge(dets, nms_iou=0.5, contain=0.8, vote=1, min_ratio=0.0, score="mean_all"):
    """메인 노트북과 같은 군집 규칙. dets: [(box, score)] 관점별 목록. score: mean_all(관점별 최고 conf 합/관점 수, 메인) / max / mean_seen"""
    B = np.concatenate([d[0] for d in dets]); S = np.concatenate([d[1] for d in dets]); V = np.concatenate([np.full(len(d[1]), i) for i, d in enumerate(dets)])
    if not len(S): return np.zeros(0)
    A = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]); reps, members = [], []
    for i in np.argsort(-S):
        if reps:
            io, ct = _iou_inter(B[i], B[reps]); ratio = np.minimum(A[i], A[reps]) / (np.maximum(A[i], A[reps]) + 1e-12)
            j = np.where((io >= nms_iou) | ((ct >= contain) & (ratio >= min_ratio)))[0]
            if len(j): members[j[0]].append(i); continue
        reps.append(i); members.append([i])
    out = []
    for mem in members:
        best = {}
        for i in mem: best[V[i]] = max(best.get(V[i], 0), S[i])
        if len(best) >= vote:
            v = list(best.values()); out.append(sum(v) / len(dets) if score == "mean_all" else (max(v) if score == "max" else sum(v) / len(v)))
    return np.array(out)


def cache(split="train", sizes=(480, 640, 800, 960, 1120, 1280)):
    fn = os.path.join(ROOT, "_exp2", f"raw_{split}.pkl")
    if os.path.exists(fn): return pickle.load(open(fn, "rb"))
    files = labels.filename if split == "train" else sample.filename; t = time.time()
    d = {f: group_views(os.path.join(DATA, f"{split}_images", f), sizes) for f in files}; pickle.dump(d, open(fn, "wb")); print(f"{split} 원시 출력 저장 {time.time() - t:.0f}s"); return d


TH = np.round(np.arange(0.05, 0.61, 0.025), 3)


def evaluate(S, y):
    """S: 이미지별 군집 점수 배열 목록. 반환: (고원 기준 임계값, 그 MAE, LOO MAE, 편향, 임계값별 MAE)"""
    E = np.array([np.abs(np.array([(s >= th).sum() for s in S]) - y) for th in TH]); pl = np.array([E[max(0, k - 1):k + 2].mean() for k in range(len(TH))]); k = int(np.argmin(pl))
    loo = np.mean([E[int(np.argmin(np.delete(E, i, 1).mean(1))), i] for i in range(len(y))]); bias = np.mean(np.array([(s >= TH[k]).sum() for s in S]) - y)
    return float(TH[k]), float(E[k].mean()), float(loo), float(bias), E.mean(1)
