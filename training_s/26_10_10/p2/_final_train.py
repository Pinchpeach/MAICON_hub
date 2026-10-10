"""노트북 셀 B, C, D 를 그대로 실행해 최종 드론 모델을 학습하고 예측을 저장한다(규칙 탐색용)."""
import pickle, sys
sys.path.insert(0, ".")
from _nb_cells_a import CELL_B, CELL_C, CELL_D
g = {}
exec(CELL_B, g); exec(CELL_C, g); exec(CELL_D, g)
pickle.dump(dict(P_drone=g["P_drone"], P_base=g["P_base"], DRONE_W=g["DRONE_W"], t_train=g["t_train"], t_pred=g["t_pred"]), open("_final_preds.pkl", "wb"))
print("DONE", g["DRONE_W"])
