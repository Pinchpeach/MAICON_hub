"""오답 이미지의 특징 분석: 최종 규칙으로 군집(박스 포함)을 만들고 이미지별 특징을 정답/오답으로 비교, 오답 이미지 시트 저장."""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "_exp")
from lib import GROUPS
from PIL import Image, ImageDraw
CFG = {("problem02", "person"): (["full960", "full1280", "tile960"], 0.5, 0.7, 2, 0.0, 0.15, 4),
       ("problem04", "person"): (["full640", "flip640", "full960", "flip960", "full1280", "flip1280"], 0.5, 0.8, 1, 0.0, 0.15, 0),
       ("problem04", "vehicle"): (["full640", "full960", "full1280"], 0.5, 0.7, 2, 0.1, 0.175, 0)}
def clusters(views, keys, group, nms, contain, vote, mr):
    B, S, V = [], [], []
    for vi, k in enumerate(keys):
        b, s, c = views[k]; m = np.isin(c, GROUPS[group]); B.append(b[m]); S.append(s[m]); V.append(np.full(int(m.sum()), vi))
    B = np.concatenate(B); S = np.concatenate(S); V = np.concatenate(V); A = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]); reps, mem = [], []
    for i in np.argsort(-S):
        if reps:
            R = B[reps]; iw = np.maximum(0, np.minimum(B[i, 2], R[:, 2]) - np.maximum(B[i, 0], R[:, 0])); ih = np.maximum(0, np.minimum(B[i, 3], R[:, 3]) - np.maximum(B[i, 1], R[:, 1])); inter = iw * ih
            io = inter / (A[i] + A[reps] - inter + 1e-12); ct = inter / (np.minimum(A[i], A[reps]) + 1e-12); ratio = np.minimum(A[i], A[reps]) / (np.maximum(A[i], A[reps]) + 1e-12)
            j = np.where((io >= nms) | ((ct >= contain) & (ratio >= mr)))[0]
            if len(j): mem[j[0]].append(i); continue
        reps.append(i); mem.append([i])
    out = []
    for r, mm in zip(reps, mem):
        best = {}
        for i in mm: best[V[i]] = max(best.get(V[i], 0), S[i])
        out.append((B[r], sum(best.values()) / len(keys), len(best)))
    return out
for (p, tgt), (keys, nms, ct, vote, mr, thr, lo) in CFG.items():
    tl = pd.read_csv(f"{p}/data/train_labels.csv"); y = tl[f"{tgt}_count"].values; V = pickle.load(open(f"_exp/{p}_views.pkl", "rb")); rows = []; keep = {}
    for f, yy in zip(tl.filename, y):
        W, H = Image.open(f"{p}/data/train_images/{f}").size; cl = clusters(V[("train_images", f)], keys, tgt, nms, ct, vote, mr)
        ok = [c for c in cl if c[2] >= vote and c[1] >= thr]; near = [c for c in cl if not (c[2] >= vote and c[1] >= thr) and c[1] >= 0.04]; keep[f] = (ok, near)
        hs = np.array([(c[0][3] - c[0][1]) * H for c in ok]) if ok else np.array([0.0]); ar = np.array([(c[0][2] - c[0][0]) * (c[0][3] - c[0][1]) for c in ok]) if ok else np.array([0.0])
        # 센 박스끼리의 최대 겹침(혼잡도)
        ov = 0.0
        for i in range(len(ok)):
            for j in range(i + 1, len(ok)):
                a, b = ok[i][0], ok[j][0]; iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1])); inter = iw * ih
                ov = max(ov, inter / (min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])) + 1e-12))
        pred = max(len(ok), lo)
        rows.append(dict(f=f, y=yy, pred=pred, err=pred - yy, n_near=len(near), n_small_near=sum(1 for c in near if (c[0][3] - c[0][1]) * H < 40), min_h=hs.min(), med_h=np.median(hs), max_area=ar.max(),
                         n_small_ok=int((hs < 40).sum()), max_overlap=ov, mean_score=np.mean([c[1] for c in ok]) if ok else 0, min_score=min([c[1] for c in ok]) if ok else 0, portrait=int(H > W)))
    D = pd.DataFrame(rows); D["abs"] = D.err.abs(); D["wrong"] = D["abs"] > 0
    print(f"\n=== {p} {tgt}: MAE {D['abs'].mean():.3f} | 오답 {int(D.wrong.sum())}장 (과소 {int((D.err < 0).sum())}, 과다 {int((D.err > 0).sum())}) | 오차 합: 과소 {int(-D.err[D.err < 0].sum())}, 과다 {int(D.err[D.err > 0].sum())}")
    cols = ["y", "pred", "n_near", "n_small_near", "n_small_ok", "min_h", "med_h", "max_area", "max_overlap", "mean_score", "min_score", "portrait"]
    g = D.groupby(D.err.map(lambda e: "과소" if e < 0 else "과다" if e > 0 else "정답"))[cols].mean().round(2); print(g.to_string())
    print("   |오차|와의 순위 상관:", {c: round(float(D[c].corr(D["abs"], method="spearman")), 2) for c in cols if c != "pred"})
    print("   정답 수량별 MAE:", D.groupby(pd.cut(D.y, [-1, 0, 2, 4, 6, 9, 20]), observed=True)["abs"].agg(["count", "mean"]).round(2).to_dict("index"))
    print(D[D.wrong].sort_values("err")[["f", "y", "pred", "err", "n_near", "n_small_near", "n_small_ok", "min_h", "max_area", "max_overlap"]].round(2).to_string(index=False))
    W_ = D[D.wrong].sort_values("err"); S = 420; n = len(W_); sheet = Image.new("RGB", (4 * S, ((n + 3) // 4) * S), (30, 30, 30))
    for j, (_, r) in enumerate(W_.iterrows()):
        im = Image.open(f"{p}/data/train_images/{r.f}").convert("RGB"); Wd, Hd = im.size; z = 2; im = im.resize((Wd * z, Hd * z)); d = ImageDraw.Draw(im); ok, near = keep[r.f]
        for c in near: d.rectangle([c[0][0] * Wd * z, c[0][1] * Hd * z, c[0][2] * Wd * z, c[0][3] * Hd * z], outline=(255, 150, 0), width=1)
        for c in ok: d.rectangle([c[0][0] * Wd * z, c[0][1] * Hd * z, c[0][2] * Wd * z, c[0][3] * Hd * z], outline=(0, 255, 0), width=3)
        d.rectangle([0, 0, 330, 24], fill=(0, 0, 0)); d.text((4, 4), f"{r.f[-8:-4]}  GT {r.y}  pred {r.pred}  ({r.err:+d})", fill=(255, 255, 0)); im.thumbnail((S - 4, S - 4)); sheet.paste(im, ((j % 4) * S, (j // 4) * S))
    sheet.save(f"_exp/err_{p}_{tgt}.jpg", quality=85); D.to_csv(f"_exp/err_{p}_{tgt}.csv", index=False)
