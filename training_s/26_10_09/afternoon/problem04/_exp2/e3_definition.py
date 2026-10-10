import sys, os, pickle; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, torchvision
from PIL import Image
import harness as H
from harness import labels, DATA, ROOT, merge, evaluate, raw_pred
files = list(labels.filename); yp = labels.person_count.values; yv = labels.vehicle_count.values
CL = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "train": 6, "truck": 7}; IDX = list(CL.values()); NAME = list(CL)
fn = os.path.join(ROOT, "_exp2", "raw_train_7cls.pkl")
if os.path.exists(fn): R = pickle.load(open(fn, "rb"))
else:
    R = {}
    for f in files:
        arr = np.asarray(Image.open(os.path.join(DATA, "train_images", f)).convert("RGB"))[:, :, ::-1]; R[f] = {}
        for s in (640, 960, 1280):
            for fl in (False, True):
                b, c = raw_pred(arr[:, ::-1].copy() if fl else arr, s)
                if fl: b = b.copy(); b[:, [0, 2]] = 1 - b[:, [2, 0]]
                R[f][(s, fl)] = (b, c[:, IDX].copy())
    pickle.dump(R, open(fn, "wb"))
def dets(view, names):
    b, c = view; s = c[:, [NAME.index(n) for n in names]].max(1); m = s >= 0.03; b, s = b[m], s[m]
    if not len(s): return np.zeros((0, 4)), np.zeros(0)
    k = torchvision.ops.nms(torch.from_numpy(b).float(), torch.from_numpy(s).float(), 0.7).numpy()[:300]; return b[k], s[k]
VV = [(s, False) for s in (640, 960, 1280)]; PV = [(s, f) for s in (640, 960, 1280) for f in (False, True)]
def veh(names): return [merge([dets(R[f][v], names) for v in VV], nms_iou=0.5, contain=0.7, vote=2, min_ratio=0.1) for f in files]
print("== 1. 차량 정의 확인: 어떤 클래스 묶음이 학습 정답과 가장 잘 맞는가 ==")
for nm, names in (("car+bus+truck (문제 정의, 메인)", ["car", "bus", "truck"]), ("+ motorcycle", ["car", "bus", "truck", "motorcycle"]), ("+ bicycle", ["car", "bus", "truck", "bicycle"]), ("+ motorcycle + bicycle", ["car", "bus", "truck", "motorcycle", "bicycle"]), ("+ train", ["car", "bus", "truck", "train"]), ("car 만", ["car"]), ("car+truck", ["car", "truck"])):
    S = veh(names); th, m, loo, b, _ = evaluate(S, yv); print("  %-32s 임계값 %.3f MAE %.3f LOO %.3f 편향 %+.2f" % (nm, th, m, loo, b))
print("\n== 2. 메인 설정의 학습 30장 성적 ==")
Sv = veh(["car", "bus", "truck"]); Sp = [merge([dets(R[f][v], ["person"]) for v in PV], nms_iou=0.5, contain=0.8, vote=1) for f in files]
pp = np.array([(s >= 0.15).sum() for s in Sp]); pv = np.array([(s >= 0.175).sum() for s in Sv])
print("  사람 MAE %.3f 완전 일치 %.3f | 차량 MAE %.3f 완전 일치 %.3f | 둘 다 맞힌 이미지 %.3f (%d/30) | 평균 MAE %.3f -> 점수 %.1f" % (np.abs(pp - yp).mean(), (pp == yp).mean(), np.abs(pv - yv).mean(), (pv == yv).mean(), ((pp == yp) & (pv == yv)).mean(), ((pp == yp) & (pv == yv)).sum(), (np.abs(pp - yp).mean() + np.abs(pv - yv).mean()) / 2, 100 / (1 + (np.abs(pp - yp).mean() + np.abs(pv - yv).mean()) / 2)))
print("\n== 3. 정답이 0 인 이미지에서의 오탐 ==")
for nm, y, p, S in (("사람", yp, pp, Sp), ("차량", yv, pv, Sv)):
    z = y == 0; print("  %s 정답 0 인 %d장: 오탐 있는 이미지 %d장 (예측 합 %d) | 정답 1 이상 %d장: 덜 셈 %d장, 더 셈 %d장" % (nm, z.sum(), (p[z] > 0).sum(), p[z].sum(), (~z).sum(), (p[~z] < y[~z]).sum(), (p[~z] > y[~z]).sum()))
    for th in (0.10, 0.15, 0.175, 0.20, 0.30): c = np.array([(s >= th).sum() for s in S]); print("     임계값 %.3f: 정답 0 이미지 오탐 %d장 / 정답 있는 이미지에서 덜 센 수량 합 %d / 전체 MAE %.3f" % (th, (c[z] > 0).sum(), np.clip(y - c, 0, None).sum(), np.abs(c - y).mean()))
print("\n== 4. S12_2_0001 (정답 사람 %d, 차량 %d) 에서 모델이 보는 것 ==" % (yp[0], yv[0]))
f = files[0]
for n in ("person", "car", "bus", "truck", "motorcycle", "bicycle"):
    S = merge([dets(R[f][v], [n]) for v in VV], nms_iou=0.5, contain=0.7, vote=1); print("  %-10s 점수 상위:" % n, np.sort(S)[::-1][:6].round(3).tolist())
print("  메인 예측: 사람 %d, 차량 %d" % (pp[0], pv[0]))
