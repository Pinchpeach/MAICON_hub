"""진단: 병합 없이 박스를 그대로 세면(기본 NMS 만) 답지 수량에 얼마나 가까운가. 답지로 훑는 진단."""
import pickle, numpy as np
AIR = 4; ev, t, P = pickle.load(open("_air_eval.pkl", "rb")); PB = pickle.load(open("_final_preds.pkl", "rb"))["P_base"]; PT = pickle.load(open("_air_tile.pkl", "rb")); multi = t >= 2
area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = iw * ih; return i / (area(a) + area(b) - i + 1e-12)
def cnt(bx, sc, thr, it):
    keep = []
    for i in np.argsort(-sc):
        if sc[i] < thr: break
        if not any(iou(bx[i], bx[j]) >= it for j in keep): keep.append(i)
    return max(len(keep), 1)
TH = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
print("%-26s " % "설정" + " ".join("%11s" % ("%.2f" % x) for x in TH) + "   (칸 = 전체 정확 / 2대 이상 정확 / 과소 수)")
for nm, get in (("640", lambda f: PB[f]), ("960", lambda f: PT[f]["s960"][:3]), ("640 TTA", lambda f: PT[f]["tta640"][:3]), ("960 TTA", lambda f: PT[f]["tta960"][:3])):
    for it in (0.5, 0.7):
        row = []
        for th in TH:
            p = np.array([cnt(get(f)[2][get(f)[0] == AIR], get(f)[1][get(f)[0] == AIR], th, it) for f in ev]); row.append("%2d/%2d/%2d" % ((p == t).sum(), ((p == t) & multi).sum(), (p < t).sum()))
        print("%-26s " % ("%s, 병합 IoU %.1f 만" % (nm, it)) + " ".join("%11s" % x for x in row))
