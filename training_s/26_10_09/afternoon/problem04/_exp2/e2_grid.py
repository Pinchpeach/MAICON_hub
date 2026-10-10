import sys, os, itertools; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from harness import *
R = cache("train"); Y = {"person": labels.person_count.values, "vehicle": labels.vehicle_count.values}; files = list(labels.filename); SZ = (480, 640, 800, 960, 1120, 1280)
BASE = {"person": dict(nms_iou=0.5, contain=0.8, min_ratio=0.0), "vehicle": dict(nms_iou=0.5, contain=0.7, min_ratio=0.1)}
MAIN = {"person": ((640, 960, 1280), True, 1, "mean_all"), "vehicle": ((640, 960, 1280), False, 2, "mean_all")}
D = {g: {f: {v: to_dets(R[f][v], g) for v in R[f]} for f in files} for g in Y}          # 관점별 NMS 결과를 미리 계산
def S_of(g, sizes, flip, vote, score):
    views = [(s, fl) for s in sizes for fl in ((False, True) if flip else (False,))]
    return [merge([D[g][f][v] for v in views], vote=vote, score=score, **BASE[g]) for f in files]
for g in ("person", "vehicle"):
    print(f"== {g}: 해상도 조합 x 반전 x 최소 관점 수 x 점수 방식 전체 탐색 ==")
    rows = []
    for r in range(1, 7):
        for sizes in itertools.combinations(SZ, r):
            for flip in (False, True):
                nv = r * (2 if flip else 1)
                for vote in (1, 2, 3):
                    if vote > nv: continue
                    for score in ("mean_all", "max", "mean_seen"):
                        th, m, loo, b, _ = evaluate(S_of(g, sizes, flip, vote, score), Y[g]); rows.append((loo, m, th, b, sizes, flip, vote, score))
    rows.sort(key=lambda x: (x[0], x[1])); loos = np.array([x[0] for x in rows]); mm = MAIN[g]
    main = next(x for x in rows if x[4] == mm[0] and x[5] == mm[1] and x[6] == mm[2] and x[7] == mm[3])
    print("  설정 %d개 | LOO MAE 분포: 최소 %.3f, 5%% %.3f, 중앙 %.3f, 95%% %.3f | 메인 LOO %.3f (순위 %d, 같은 값 이하 %d개)" % (len(rows), loos.min(), np.percentile(loos, 5), np.median(loos), np.percentile(loos, 95), main[0], int((loos < main[0]).sum()) + 1, int((loos <= main[0]).sum())))
    for x in rows[:6]: print("   LOO %.3f MAE %.3f 임계값 %.3f 편향 %+.2f | 해상도 %s 반전 %s 최소관점 %d 점수 %s" % x)
    for sc in ("mean_all", "max", "mean_seen"): print("   점수 방식 %-9s LOO 중앙값 %.3f, 최소 %.3f" % (sc, np.median([x[0] for x in rows if x[7] == sc]), min(x[0] for x in rows if x[7] == sc)))
    for fl in (False, True): print("   반전 %-5s LOO 중앙값 %.3f" % (fl, np.median([x[0] for x in rows if x[5] == fl])))
    for n in range(1, 7): print("   해상도 %d개 LOO 중앙값 %.3f, 최소 %.3f" % (n, np.median([x[0] for x in rows if len(x[4]) == n]), min(x[0] for x in rows if len(x[4]) == n)))
    # ----- 밀도 적응 임계값과 점수 합 계수 (메인 관점 설정에서) -----
    S = S_of(g, *mm); y = Y[g]; th0, m0, loo0, _, _ = evaluate(S, y)
    def loo_param(pred_fn, grid):
        E = np.array([[abs(pred_fn(s, *p) - yy) for s, yy in zip(S, y)] for p in grid]); k = int(np.argmin(E.mean(1)))
        return E[k].mean(), np.mean([E[int(np.argmin(np.delete(E, i, 1).mean(1))), i] for i in range(len(y))]), grid[k]
    dens = lambda s, t, d, k, hi: int((s >= (t - d if (s >= hi).sum() >= k else t)).sum())
    g1 = [(t, d, k, hi) for t in (0.125, 0.15, 0.175, 0.2) for d in (0.025, 0.05, 0.075) for k in (2, 3, 4, 5) for hi in (0.3, 0.5)]
    m1, l1, p1 = loo_param(dens, g1)
    soft = lambda s, lo, ref: int(np.round(np.minimum(1.0, s[s >= lo] / ref).sum()))
    g2 = [(lo, ref) for lo in (0.05, 0.075, 0.1, 0.125, 0.15) for ref in (0.15, 0.2, 0.25, 0.3, 0.4)]
    m2, l2, p2 = loo_param(soft, g2)
    print("  메인 관점에서: 단일 임계값 MAE %.3f / LOO %.3f | 밀도 적응 임계값 MAE %.3f / LOO %.3f %s | 점수 합 계수 MAE %.3f / LOO %.3f %s\n" % (m0, loo0, m1, l1, p1, m2, l2, p2))
