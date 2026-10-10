"""사용: python _train_tl.py <가중치 이름 n|s> <실험 이름> <epochs> <imgsz> <batch> [data yaml] [추가 인자 json]"""
import sys, os, time, json, warnings; warnings.filterwarnings("ignore")
import torch
from ultralytics import YOLO
W = {"n": r"C:\Users\User\Desktop\root\docker11\notebooks\MAICON_hub\training_s\26_10_09\all_data\problem01\yolov8n.pt", "s": r"C:\Users\User\Desktop\root\docker11\notebooks\MAICON_hub\training_s\26_10_09\all_data\problem01\yolov8s.pt"}
size, name, epochs, imgsz, batch = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
data = sys.argv[6] if len(sys.argv) > 6 else "_tl_data.yaml"; extra = json.loads(sys.argv[7]) if len(sys.argv) > 7 else {}
dev = 0 if torch.cuda.is_available() else "cpu"; t0 = time.time()
kw = dict(patience=0, seed=0, deterministic=True, plots=False, verbose=False, workers=0, exist_ok=True); kw.update(extra)
m = YOLO(W[size]); m.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=dev, project=os.path.abspath("runs_tl"), name=name, **kw)
def ev(wp):
    r = YOLO(wp).val(data="_tl_data.yaml", imgsz=imgsz, device=dev, plots=False, verbose=False, workers=0, batch=8)
    return dict(map50=round(float(r.box.map50), 4), map=round(float(r.box.map), 4), p=round(float(r.box.mp), 4), r=round(float(r.box.mr), 4))
sd = str(m.trainer.save_dir); sec = round(time.time() - t0, 1)
last, best = ev(os.path.join(sd, "weights", "last.pt")), ev(os.path.join(sd, "weights", "best.pt"))
print("RESULT", json.dumps(dict(name=name, size=size, epochs=epochs, imgsz=imgsz, batch=batch, extra=extra, seconds=sec, sec_per_epoch=round(sec / epochs, 2), last=last, best=best)), flush=True)
