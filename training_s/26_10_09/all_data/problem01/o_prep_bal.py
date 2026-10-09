"""⑤ 소수 클래스(truck, military_vehicle, civilian_vehicle)가 있는 학습 이미지를 반복한 목록과 data yaml 생성. 원본 data/ 는 건드리지 않는다."""
import glob, os, collections

REPEAT = 3
root = os.path.abspath("data").replace("\\", "/")
lines, cnt = [], collections.Counter()
for img in sorted(glob.glob("data/train/images/*")):
    lab = img.replace("\\", "/").replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    cls = {int(l.split()[0]) for l in open(lab).read().splitlines() if l.strip()}
    k = REPEAT if cls & {1, 2, 3} else 1
    lines += [os.path.abspath(img).replace("\\", "/")] * k
    for c in cls: cnt[c] += k
open("o_train_bal.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
open("data_o_bal.yaml", "w", encoding="utf-8").write(
    f"path: {root}\ntrain: {os.path.abspath('o_train_bal.txt').replace(chr(92), '/')}\nval: val/images\n"
    "names:\n  0: military_tank\n  1: military_truck\n  2: military_vehicle\n  3: civilian_vehicle\n")
print("학습 목록 행 수:", len(lines), "| 클래스별 이미지 수(반복 포함):", dict(sorted(cnt.items())))
