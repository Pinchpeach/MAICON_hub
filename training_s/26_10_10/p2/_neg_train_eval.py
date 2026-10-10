"""부정 예시를 더한 드론 모델 학습(겹 0)과 임계값 선택. 사용: python _neg_train_eval.py"""
import os, json, glob, time, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from ultralytics import YOLO
dev = 0 if torch.cuda.is_available() else "cpu"; yml = os.path.abspath("_neg_set0/data.yaml"); t0 = time.time()
m = YOLO("yolov8s.pt"); m.train(data=yml, epochs=20, imgsz=640, batch=8, device=dev, optimizer="AdamW", lr0=0.0002, patience=0, seed=0, deterministic=True, workers=0, plots=False, verbose=False,
                                exist_ok=True, cache="ram", project=os.path.abspath("runs_neg"), name="s_neg_f0")
print("학습 %.0fs" % (time.time() - t0), flush=True)
W = os.path.join(str(m.trainer.save_dir), "weights", "last.pt"); lm = YOLO(W)
def maxconf(files):
    out = []
    for i in range(0, len(files), 8):
        for r in lm.predict(files[i:i + 8], imgsz=640, conf=0.01, iou=0.6, device=dev, verbose=False): out.append(float(r.boxes.conf.max()) if len(r.boxes) else 0.0)
    return np.array(out)
vd = sorted(glob.glob("_neg_set0/val/images/*.jpg")); vd_pos = [f for f in vd if not os.path.basename(f).startswith("neg_")]; vd_neg = [f for f in vd if os.path.basename(f).startswith("neg_")]
sp, sn = maxconf(vd_pos), maxconf(vd_neg); kinds = json.load(open("_neg_split.json"))["kinds"]
nk = np.array([kinds[os.path.basename(f)[4:]] for f in vd_neg]); tn = maxconf(sorted(glob.glob("_neg_set0/train/images/neg_*.jpg")))
print("== 보류 검증: 드론 %d장 대 부정 %d장 (새 %d, 비행기 %d) ==" % (len(sp), len(sn), (nk == "bird").sum(), (nk == "airplane").sum()))
print("  드론 점수 분위(10, 25, 50, 75, 90): %s" % np.percentile(sp, [10, 25, 50, 75, 90]).round(2).tolist())
print("  부정 점수 분위(50, 75, 90, 100): %s | 새 %s | 비행기 %s" % (np.percentile(sn, [50, 75, 90, 100]).round(2).tolist(), np.percentile(sn[nk == "bird"], [50, 90, 100]).round(2).tolist(), np.percentile(sn[nk == "airplane"], [50, 90, 100]).round(2).tolist()))
print("  (참고) 학습에 쓴 부정 %d장의 점수 분위(50, 90, 100): %s" % (len(tn), np.percentile(tn, [50, 90, 100]).round(2).tolist()))
print("  임계값별: 드론 재현율 / 부정 오탐률(새·비행기 따로)")
for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8): print("    %.1f: 드론 %.2f | 부정 %.2f (새 %.2f, 비행기 %.2f)" % (t, (sp >= t).mean(), (sn >= t).mean(), (sn[nk == "bird"] >= t).mean(), (sn[nk == "airplane"] >= t).mean()))
pickle.dump(dict(W=W, sp=sp, sn=sn, nk=nk), open("_neg_eval.pkl", "wb")); print("WEIGHTS", W)
