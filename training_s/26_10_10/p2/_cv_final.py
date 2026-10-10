import json, collections, glob, os, warnings, sys; warnings.filterwarnings("ignore")
import numpy as np
R = [json.loads(l) for l in open("_cv_results.jsonl", encoding="utf-8") if "s_e20_" in l]
tag = lambda r: r["tag"].replace("s_e20_", "").replace("_lr0.0002", "")
G = collections.defaultdict(dict)
for r in R: G[tag(r)][r["fold"]] = r
order = ["base", "dihedral", "synth", "dihedral+synth"]
print("== 겹 0~2 확정 결과 (yolov8s, lr 0.0002, 20에폭; 겹당 검증 이미지 약 12장 + 합성 24장) ==")
print("%-16s %-8s %-8s %-9s %-12s %-12s %-9s" % ("구성", "mAP50", "재현율", "단일정확", "다중정확", "다중오차", "학습이미지"))
for c in order:
    v = [G[c][k] for k in (0, 1, 2)]; m = lambda key: np.mean([x[key] for x in v]); s = lambda key: np.std([x[key] for x in v])
    print("%-16s %-8.3f %-8.3f %-9.3f %-5.3f±%-5.3f %-12.3f %d" % (c, m("map50"), m("recall"), m("single_exact"), m("multi_exact"), s("multi_exact"), m("multi_mae"), m("train_images")))
print("\n겹별 다중 개수 정확도 / 단일 개수 정확도:")
for k in (0, 1, 2): print("  fold %d: " % k + " | ".join("%s %.2f/%.2f" % (c, G[c][k]["multi_exact"], G[c][k]["single_exact"]) for c in order))
b = order[3]; print("\n둘 다(dihedral+synth) - 기준, 겹별 다중 정확도 차:", [round(G[b][k]["multi_exact"] - G["base"][k]["multi_exact"], 2) for k in (0, 1, 2)], "| 단일 정확도 차:", [round(G[b][k]["single_exact"] - G["base"][k]["single_exact"], 2) for k in (0, 1, 2)])
print("둘 다 - 합성 단독, 다중 차:", [round(G[b][k]["multi_exact"] - G["synth"][k]["multi_exact"], 2) for k in (0, 1, 2)], "| 단일 차:", [round(G[b][k]["single_exact"] - G["synth"][k]["single_exact"], 2) for k in (0, 1, 2)])
# 임계값 선택: 겹 0~2 의 'dihedral+synth' 모델로 원본 검증 이미지(단일)와 합성 검증 이미지(다중)의 개수 정확도를 임계값별로 본다
import torch; from ultralytics import YOLO
dev = 0 if torch.cuda.is_available() else "cpu"; TH = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7); tot = {t: [0, 0, 0, 0] for t in TH}   # 단일 정답수, 단일 n, 다중 정답수, 다중 n
for k in (0, 1, 2):
    w = f"runs_cv/s_e20_dihedral+synth_lr0.0002_f{k}/weights/last.pt"; m = YOLO(w)
    for d, ix in ((f"_cv/dihedral+synth/f{k}/val", 0), (f"_cv/synthval/f{k}", 2)):
        fs = sorted(glob.glob(d + "/images/*.jpg"))
        for i in range(0, len(fs), 8):
            for p, r in zip(fs[i:i + 8], m.predict(fs[i:i + 8], imgsz=640, conf=0.05, iou=0.6, device=dev, verbose=False)):
                n = len([l for l in open(os.path.join(d, "labels", os.path.splitext(os.path.basename(p))[0] + ".txt")).read().splitlines() if l.strip()])
                for t in TH: tot[t][ix + 1] += 1; tot[t][ix] += int(int((r.boxes.conf.cpu().numpy() >= t).sum()) == n)
print("\n== 드론 임계값별 개수 정확도 (둘 다 모델, 겹 0~2 검증 36장 / 합성 72장) ==")
for t in TH: print("  conf %.1f: 단일 %.3f (%d/%d) | 다중 %.3f (%d/%d)" % (t, tot[t][0] / tot[t][1], tot[t][0], tot[t][1], tot[t][2] / tot[t][3], tot[t][2], tot[t][3]))
