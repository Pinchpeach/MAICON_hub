"""직접 고친 박스 라벨로 yolov8n 미세조정 -> 5겹 OOF 로 수량 MAE 평가. 사용: python ft_cv.py <변형 이름> <라벨 폴더> <epochs> <freeze> [lr0]"""
import sys, os, shutil, time, json, pickle, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, torch
from PIL import Image
from ultralytics import YOLO
import harness as H
from harness import labels, DATA, ROOT, group_views, to_dets, merge, evaluate

name, lab_dir, epochs, freeze = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); lr0 = float(sys.argv[5]) if len(sys.argv) > 5 else None
EXP = os.path.join(ROOT, "_exp2", "ft", name); files = list(labels.filename); SUB = {0: 2, 1: 5, 2: 7}            # car / bus / truck -> COCO id
TRUCK_HAND = {("S12_2_0016.jpg", (0.61, 0.42))}                                                                     # 직접 그린 박스 중 트럭


def coco_labels():
    """폴더의 2클래스 라벨(0 사람, 1 차량)을 COCO 번호(0, 2/5/7)로 바꾼다. 차량 세부 클래스는 모델 군집과 겹치면 그 클래스, 아니면 승용차"""
    import make_dummy as M
    out = {}
    for f in files:
        rows = [l.split() for l in open(os.path.join(lab_dir, f[:-4] + ".txt")).read().splitlines() if l.strip()]; cl = M.clusters(f, "vehicle"); lines = []
        for c, xc, yc, w, h in rows:
            xc, yc, w, h = map(float, (xc, yc, w, h))
            if c == "0": cid = 0
            else:
                b = np.array([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]); cid = 2
                for cb, s, k, nv in cl:
                    iw = max(0, min(b[2], cb[2]) - max(b[0], cb[0])); ih = max(0, min(b[3], cb[3]) - max(b[1], cb[1])); inter = iw * ih
                    if inter / ((b[2] - b[0]) * (b[3] - b[1]) + (cb[2] - cb[0]) * (cb[3] - cb[1]) - inter + 1e-12) > 0.9: cid = SUB[k]; break
                if any(f == tf and abs(b[0] - x0) < 0.01 and abs(b[1] - y0) < 0.01 for tf, (x0, y0) in TRUCK_HAND): cid = 7
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        out[f] = lines
    return out


if __name__ == "__main__":
    os.makedirs(EXP, exist_ok=True); L = coco_labels(); names = YOLO(os.path.join(ROOT, "yolov8n.pt")).names
    rng = np.random.RandomState(0); perm = rng.permutation(30); folds = [perm[i::5] for i in range(5)]; RAW = {}; t0 = time.time()
    for k, va in enumerate(folds):
        d = os.path.join(EXP, f"fold{k}"); shutil.rmtree(d, ignore_errors=True)
        for sp in ("images/train", "labels/train"): os.makedirs(os.path.join(d, sp))
        for i in range(30):
            if i in va: continue
            f = files[i]; shutil.copy(os.path.join(DATA, "train_images", f), os.path.join(d, "images/train", f)); open(os.path.join(d, "labels/train", f[:-4] + ".txt"), "w").write("\n".join(L[f]) + ("\n" if L[f] else ""))
        yml = os.path.join(d, "data.yaml"); open(yml, "w", encoding="utf-8").write(f"path: {json.dumps(d)}\ntrain: images/train\nval: images/train\nnames:\n" + "".join(f"  {i}: {json.dumps(n)}\n" for i, n in names.items()))
        kw = dict(lr0=lr0, optimizer="AdamW") if lr0 else {}
        m = YOLO(os.path.join(ROOT, "yolov8n.pt")); m.train(data=yml, epochs=epochs, imgsz=640, batch=8, workers=0, device=0 if torch.cuda.is_available() else "cpu", freeze=freeze, val=False, plots=False, verbose=False,
                                                         seed=0, deterministic=True, project=os.path.join(EXP, "runs"), name=f"fold{k}", exist_ok=True, patience=0, **kw)
        w = os.path.join(str(m.trainer.save_dir), "weights", "last.pt"); H._model = YOLO(w); H._model.model.eval().cpu().float()
        for i in va: RAW[files[i]] = group_views(os.path.join(DATA, "train_images", files[i]), sizes=(640, 960, 1280))
        print(f"[{name}] fold {k} 완료 ({time.time() - t0:.0f}s)", flush=True)
    pickle.dump(RAW, open(os.path.join(EXP, "oof_raw.pkl"), "wb"))
    PV = [(s, f) for s in (640, 960, 1280) for f in (False, True)]; VV = [(s, False) for s in (640, 960, 1280)]; yp, yv = labels.person_count.values, labels.vehicle_count.values
    Sp = [merge([to_dets(RAW[f][v], "person") for v in PV], nms_iou=0.5, contain=0.8, vote=1) for f in files]; Sv = [merge([to_dets(RAW[f][v], "vehicle") for v in VV], nms_iou=0.5, contain=0.7, vote=2, min_ratio=0.1) for f in files]
    rp, rv = evaluate(Sp, yp), evaluate(Sv, yv)
    fixp = np.abs(np.array([(s >= 0.15).sum() for s in Sp]) - yp).mean(); fixv = np.abs(np.array([(s >= 0.175).sum() for s in Sv]) - yv).mean()
    res = dict(name=name, epochs=epochs, freeze=freeze, person_thr=rp[0], person_mae=rp[1], person_loo=rp[2], person_bias=rp[3], vehicle_thr=rv[0], vehicle_mae=rv[1], vehicle_loo=rv[2], vehicle_bias=rv[3], person_mae_mainthr=float(fixp), vehicle_mae_mainthr=float(fixv), seconds=time.time() - t0)
    json.dump(res, open(os.path.join(EXP, "result.json"), "w"), indent=1)
    print("RESULT", json.dumps(res, ensure_ascii=False), flush=True)
    for k in range(5): shutil.rmtree(os.path.join(EXP, f"fold{k}"), ignore_errors=True)                           # 복사한 학습 이미지 정리 (가중치는 runs/ 에 남김)
