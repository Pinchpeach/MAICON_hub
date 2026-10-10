"""LLM(Claude)이 후보 번호를 보고 고른 결과를 박스 라벨로 만든다. 정답 수량과 교차검증해 일치하는 이미지만 '검증됨'으로 표시.
keep = 사람으로 판단한 후보 번호, manual = 직접 그린 박스(원본 픽셀 x1,y1,x2,y2), fill = (영역 x1,y1,x2,y2, 개수): 영역 안 후보를 확신도순으로 그 개수만큼(밀집 구역)."""
import json, numpy as np, pandas as pd
from PIL import Image
p = "problem02"; tl = pd.read_csv(f"{p}/data/train_labels.csv").set_index("filename"); C = json.load(open("_exp/llm/cands.json"))
REVIEW = {
 "0008": dict(keep=[0, 1, 2, 3, 4, 5]),                                                    # 수평선 물속의 6명
 "0029": dict(keep=[1, 3, 4, 13, 10, 6, 12], manual=[(140, 118, 192, 312)]),               # 서 있는 여성은 상·하체 박스가 나뉘어 직접 그림
 "0011": dict(keep=[0, 1, 2, 3], manual=[(330, 170, 440, 360), (432, 128, 458, 170)]),     # 가운데 검은 머리 여성, 뒤쪽 머리
 "0004": dict(keep=[7, 9, 2, 5, 1, 6, 3, 4]),                                              # 0번은 9번과 같은 사람(머리)이라 제외
 "0007": dict(keep=[0, 1, 3], fill=[(400, 285, 535, 372, 10)]),                            # 오른쪽 인도 군중은 영역 지정, 왼쪽 후보는 교통콘
 "0030": dict(keep=[0, 1, 3, 6], fill=[(180, 255, 245, 305, 6), (575, 268, 612, 290, 3)]), # 보트 위 사람들, 부표는 제외
 "0020": dict(keep=[0, 6, 9, 3, 4, 2, 8, 5, 12]),
 "0013": dict(keep=[1, 4, 3], manual=[(405, 120, 525, 235), (385, 188, 425, 225)]),        # 소년(머리·몸 박스 분리), 소녀 뒤에 가려진 머리
 "0018": dict(keep=[0, 1, 2, 3], manual=[(212, 190, 290, 300), (208, 268, 250, 345)]),     # 갈색 셔츠 남자, 빨간 옷 아이
 "0021": dict(keep=[1, 5], manual=[(545, 65, 640, 420), (622, 265, 640, 420)]),            # 금발 여성(상·하체 분리), 오른쪽 가장자리 사람. 벽화 인물(3)은 제외
 "0016": dict(keep=[0, 1, 2, 3, 5, 9]),
 "0001": None, "0010": None, "0003": None,                                                 # 검토했지만 정답 수량과 맞출 수 없음(가려짐·잘림) → 학습 제외
}
out = {}; rows = []
for f in tl.index:
    k = f[-8:-4]; W, H = Image.open(f"{p}/data/train_images/{f}").size; cs = C[f]; y = int(tl.person_count[f]); src = "auto"
    if k in REVIEW:
        r = REVIEW[k]; src = "llm"
        if r is None: out[f] = dict(boxes=[], valid=False, src="llm_unresolved"); rows.append((k, y, 0, False, "llm_unresolved")); continue
        boxes = [cs[i]["box"] for i in r.get("keep", [])] + [[x1 / W, y1 / H, x2 / W, y2 / H] for x1, y1, x2, y2 in r.get("manual", [])]; used = set(r.get("keep", []))
        for x1, y1, x2, y2, n in r.get("fill", []):
            inreg = [(c["conf"], i) for i, c in enumerate(cs) if i not in used and x1 <= (c["box"][0] + c["box"][2]) / 2 * W <= x2 and y1 <= (c["box"][1] + c["box"][3]) / 2 * H <= y2]
            for _, i in sorted(inreg, reverse=True)[:n]: boxes.append(cs[i]["box"]); used.add(i)
    else:
        boxes = [c["box"] for c in cs if c["counted"]]
    ok = len(boxes) == y; out[f] = dict(boxes=boxes, valid=bool(ok), src=src); rows.append((k, y, len(boxes), ok, src))
D = pd.DataFrame(rows, columns=["img", "정답", "박스수", "일치", "출처"]); json.dump(out, open("_exp/llm/labels.json", "w"))
print(D.groupby("출처").agg(이미지=("img", "size"), 일치=("일치", "sum")).to_string()); print("불일치:", D[~D.일치].to_dict("records"))
print("검증된 이미지", int(D.일치.sum()), "/ 30 | 검증된 박스 합", int(D[D.일치].박스수.sum()))
