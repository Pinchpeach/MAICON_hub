"""사용: python _run_cv.py <모델 n|s> <epochs> <구성: base|dihedral|synth|dihedral+synth> [fold 목록 0,1,2,3,4] [lr0]
결과: _cv_results.jsonl 에 한 줄씩 추가."""
import sys, os, json, time, glob, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from ultralytics import YOLO
import _cvlib as C
W = {"n": r"C:\Users\User\Desktop\root\docker11\notebooks\MAICON_hub\training_s\26_10_09\all_data\problem01\yolov8n.pt", "s": r"C:\Users\User\Desktop\root\docker11\notebooks\MAICON_hub\training_s\26_10_09\all_data\problem01\yolov8s.pt"}
size, epochs, cfgname = sys.argv[1], int(sys.argv[2]), sys.argv[3]; fl = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 1, 2, 3, 4]
lr0 = float(sys.argv[5]) if len(sys.argv) > 5 else None; cfg = set() if cfgname == "base" else set(cfgname.split("+")); folds = json.load(open("_folds.json"))["folds"]
dev = 0 if torch.cuda.is_available() else "cpu"; tag = f"{size}_e{epochs}_{cfgname}" + (f"_lr{lr0}" if lr0 else "")
def count_eval(m, d, thr=0.25):
    fs = sorted(glob.glob(f"{d}/images/*.jpg")); err, ok, tot = [], 0, 0
    for i in range(0, len(fs), 8):
        for p, r in zip(fs[i:i + 8], m.predict(fs[i:i + 8], imgsz=640, conf=thr, iou=0.6, device=dev, verbose=False)):
            lab = os.path.join(d, "labels", os.path.splitext(os.path.basename(p))[0] + ".txt"); n = len([l for l in open(lab).read().splitlines() if l.strip()]); pn = len(r.boxes)
            err.append(abs(pn - n)); ok += int(pn == n); tot += 1
    return dict(exact=ok / max(tot, 1), mae=float(np.mean(err)), n=tot)
for k in fl:
    t0 = time.time(); yml = C.build_fold(cfg, k, folds); sv = C.build_synth_val(k, folds); ntr = len(glob.glob(os.path.dirname(yml) + "/train/images/*"))
    m = YOLO(W[size]); kw = dict(patience=0, seed=0, deterministic=True, plots=False, verbose=False, workers=0, exist_ok=True, cache="ram"); kw.update(dict(optimizer="AdamW", lr0=lr0) if lr0 else {})
    m.train(data=yml, epochs=epochs, imgsz=640, batch=8, device=dev, project=os.path.abspath("runs_cv"), name=f"{tag}_f{k}", **kw)
    lm = YOLO(os.path.join(str(m.trainer.save_dir), "weights", "last.pt")); r = lm.val(data=yml, imgsz=640, device=dev, plots=False, verbose=False, workers=0, batch=8)
    single = count_eval(lm, os.path.dirname(yml) + "/val"); multi = count_eval(lm, sv)
    res = dict(tag=tag, fold=k, train_images=ntr, map50=round(float(r.box.map50), 4), recall=round(float(r.box.mr), 4), single_exact=round(single["exact"], 3), multi_exact=round(multi["exact"], 3), multi_mae=round(multi["mae"], 3), multi_n=multi["n"], seconds=round(time.time() - t0, 1))
    open("_cv_results.jsonl", "a", encoding="utf-8").write(json.dumps(res, ensure_ascii=False) + "\n"); print("RESULT", json.dumps(res, ensure_ascii=False), flush=True)
