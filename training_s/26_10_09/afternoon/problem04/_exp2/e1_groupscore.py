import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from harness import *
R = cache("train"); y_p = labels.person_count.values; y_v = labels.vehicle_count.values; files = list(labels.filename)
def run(group, views, combine="max", **mk):
    S = [merge([to_dets(R[f][v], group, combine) for v in views], **mk) for f in files]; return evaluate(S, y_p if group == "person" else y_v), S
PV = [(s, f) for s in (640, 960, 1280) for f in (False, True)]; VV = [(s, False) for s in (640, 960, 1280)]
print("== 0. 재현 확인 (메인: person 0.333 / LOO 0.333, vehicle 0.433 / LOO 0.467) ==")
(th, m, loo, b, _), _ = run("person", PV, nms_iou=0.5, contain=0.8, vote=1); print("  person  직접 추론: 임계값 %.3f MAE %.3f LOO %.3f 편향 %+.2f" % (th, m, loo, b))
(th, m, loo, b, _), S0 = run("vehicle", VV, "max", nms_iou=0.5, contain=0.7, vote=2, min_ratio=0.1); print("  vehicle 직접 추론(max): 임계값 %.3f MAE %.3f LOO %.3f 편향 %+.2f" % (th, m, loo, b))
print("\n== 1. 차량 점수 결합 방식 (병합 규칙은 메인과 동일: 포함도 0.7, 2관점 이상, 면적비 0.1) ==")
res = {}
for comb in ("max", "sum", "nor"):
    for views, vn in ((VV, "3해상도"), (PV, "3해상도+반전")):
        for vote in (1, 2, 3):
            (th, m, loo, b, curve), S = run("vehicle", views, comb, nms_iou=0.5, contain=0.7, vote=vote, min_ratio=0.1); res[(comb, vn, vote)] = (m, loo, S, th)
            print("  결합 %-3s %-10s 최소관점 %d | 임계값 %.3f MAE %.3f LOO %.3f 편향 %+.2f" % (comb, vn, vote, th, m, loo, b))
print("\n== 2. 이미지별 변화: max -> sum (3해상도, 2관점 이상) ==")
a, bb = res[("max", "3해상도", 2)], res[("sum", "3해상도", 2)]
pa = np.array([(s >= a[3]).sum() for s in a[2]]); pb = np.array([(s >= bb[3]).sum() for s in bb[2]])
for i, f in enumerate(files):
    if pa[i] != pb[i]: print("  %s 정답 %d | max %d -> sum %d  (%s)" % (f, y_v[i], pa[i], pb[i], "개선" if abs(pb[i] - y_v[i]) < abs(pa[i] - y_v[i]) else ("악화" if abs(pb[i] - y_v[i]) > abs(pa[i] - y_v[i]) else "같음")))
print("  임계값: max %.3f, sum %.3f | 오차 합 max %d, sum %d" % (a[3], bb[3], np.abs(pa - y_v).sum(), np.abs(pb - y_v).sum()))
