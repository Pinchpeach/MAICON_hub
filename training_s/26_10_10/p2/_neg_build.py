"""테스트 이미지 중 기본 모델이 새·비행기로 확신하는 것을 '드론 없음' 부정 예시(빈 라벨)로 만들어 겹 0 학습 세트에 더한다.
사용 범위: 테스트 이미지의 입력만 사용, 정답 미사용. 가짜 라벨은 기본 yolov8s 의 확신 예측."""
import os, sys, json, random, shutil, pickle
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, ".")
import _cvlib as C
AIR, BIRD = 4, 14; AO = "data/aerial_objects"; folds = json.load(open("_folds.json"))["folds"]
P = pickle.load(open("_final_preds.pkl", "rb"))["P_base"]; rows = []
for f, (c, s, b) in P.items():
    sa = s[c == AIR].max() if (c == AIR).any() else 0.0; sb = s[c == BIRD].max() if (c == BIRD).any() else 0.0; ar = np.nan
    if sa > 0:
        i = np.where(c == AIR)[0][np.argmax(s[c == AIR])]; W, H = Image.open(os.path.join(AO, f)).size; ar = (b[i][2] - b[i][0]) * W / max((b[i][3] - b[i][1]) * H, 1e-6)
    rows.append(dict(f=f, sa=sa, sb=sb, ar=ar))
T = pd.DataFrame(rows)
nb = T[(T.sb >= 0.7) & (T.sa < 0.3)]; na = T[(T.sa >= 0.8) & (T.ar >= 3.0) & (T.sb < 0.3)]
print("부정 예시 후보: 새(sb>=0.7, sa<0.3) %d장, 비행기(sa>=0.8, 가로/세로>=3.0, sb<0.3) %d장" % (len(nb), len(na)))
rng = random.Random(0); neg = [(f, "bird") for f in nb.f] + [(f, "airplane") for f in na.f]; rng.shuffle(neg); nh = max(1, int(0.2 * len(neg))); hold, trn = neg[:nh], neg[nh:]
print("부정 예시: 학습 %d장, 보류(임계값 선택용) %d장" % (len(trn), len(hold)))
yml = C.build_fold({"dihedral", "synth"}, 0, folds)          # 겹 0: 학습 드론 46장 + 회전·반전 + 합성, 검증 드론 12장
base = os.path.dirname(yml); NEG = "_neg_set0"; shutil.rmtree(NEG, ignore_errors=True); shutil.copytree(base, NEG)
def put(f, d):
    im = Image.open(os.path.join(AO, f)).convert("RGB"); w, h = im.size; s = 640 / max(w, h)
    if s < 1: im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    os.makedirs(f"{NEG}/{d}/images", exist_ok=True); os.makedirs(f"{NEG}/{d}/labels", exist_ok=True)
    stem = "neg_" + os.path.splitext(f)[0]; im.save(f"{NEG}/{d}/images/{stem}.jpg", quality=92); open(f"{NEG}/{d}/labels/{stem}.txt", "w").write("")
for f, k in trn: put(f, "train")
for f, k in hold: put(f, "val")
open(f"{NEG}/data.yaml", "w", encoding="utf-8").write("path: %s\ntrain: train/images\nval: val/images\nnames:\n  0: drone\n" % json.dumps(os.path.abspath(NEG)))
json.dump(dict(train=[f for f, k in trn], hold=[f for f, k in hold], kinds=dict(neg)), open("_neg_split.json", "w"))
n_pos = len([x for x in os.listdir(f"{NEG}/train/images") if not x.startswith("neg_")]); print("학습 세트: 드론 이미지 %d장(회전·합성 포함) + 부정 %d장 | 검증 세트: 드론 %d장 + 부정 %d장" % (n_pos, len(trn), len(os.listdir(f"{base}/val/images")), len(hold)))
