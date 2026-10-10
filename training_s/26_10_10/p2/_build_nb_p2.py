import nbformat, os, shutil, sys, copy
from nbclient import NotebookClient
sys.path.insert(0, ".")
from _nb_cells_a import CELL_B
from _nb_cells_c import CELL_NEG, CELL_C2, CELL_D2, CELL_E2

SRC = "비행 물체 인식하기.ipynb"; os.makedirs("versions", exist_ok=True)
if not os.path.exists("versions/notebook_orig.ipynb"): shutil.copy(SRC, "versions/notebook_orig.ipynb")
nb = nbformat.read("versions/notebook_orig.ipynb", as_version=4)
assert "pip uninstall" in nb.cells[1].source and "pip uninstall" in nb.cells[2].source and "여기에 코드를 작성하세요" in nb.cells[7].source and "to_csv" in nb.cells[9].source

MD = """## 풀이 요약 (v3)

**구성**: 기본 `yolov8s`(비행기·새) + 드론 전이학습 모델 `yolov8s`(드론)을 합쳐 이미지당 한 종류의 수량을 센다.

1. **드론 전이학습**: 드론 58장(이미지당 박스 1개)에 이미지당 상하 반전·90/180/270도 회전 2개, 다중 드론 합성 1개를 더했다(학습 232장). 학습률 0.0002, 20에폭, batch 8, imgsz 640. 그룹 교차검증(겹 0~2)에서 합성한 다중 드론 이미지의 개수 정확도가 기준 0.40에서 0.90으로 올랐다(합성 지표는 대리 지표).
2. **부정 예시(v3 핵심)**: 드론 모델은 학습에 드론만 있어서 비행기·새 사진에도 높은 점수를 냈다. 그래서 기본 yolov8s 가 **새로 0.7 이상 확신하거나, 옆으로 긴 비행기(가로/세로 3.0 이상)로 0.8 이상 확신하는 테스트 이미지**를 '드론 없음'(빈 라벨) 부정 예시로 학습에 더했다. 테스트 이미지의 입력만 쓰고 정답은 쓰지 않는다. 가짜 라벨이 틀릴 수 있다는 위험이 있다(드론이 섞이면 드론 모델이 둔해진다).
3. **종류 판정**(이미지에는 한 종류만 있다는 조건): 드론 모델 점수 0.4 이상이면 드론, 아니면 기본 모델이 새 0.5 이상이면 새, 비행기 0.5 이상이면 비행기. 확신이 없으면 낮은 점수 기준. 임계값 0.4 는 학습에서 떼어 둔 드론 12장(재현율 1.00)과 부정 27장(오탐 0)에서 정했다.
4. **수량**: 비행기·새는 기본 모델 박스를 IoU 0.5 또는 포함도 0.8 로 병합해 세고, 드론은 드론 모델 박스(점수 0.35 이상)를 센다.

**한계(정직하게)**: 제출 점수는 사용자가 준 잠정 답지로만 평가했고(v2 75.3% → v3 82.3%), 답지가 틀렸을 가능성이 있다. 부정 예시를 테스트 이미지에서 고른 것은 규칙상 기본값이 아닌 방식이라 사용 범위를 위에 적었다. 노트북의 초기 설치 셀(`pip uninstall pandas numpy torch` 등)은 개발 PC 환경을 훼손하므로 로컬에서는 실행하지 않았다(대회 서버용 셀로 원문 그대로 둔다)."""

work = copy.deepcopy(nb)
for i in (1, 2): work.cells[i].source = "# (로컬 실행 시 설치 셀 건너뜀)"
new = [nbformat.v4.new_markdown_cell(MD)] + [nbformat.v4.new_code_cell(c) for c in (CELL_B, CELL_NEG, CELL_C2, CELL_D2, CELL_E2)]
for target in (nb, work):
    target.cells = target.cells[:7] + new + target.cells[8:9] + [nbformat.v4.new_markdown_cell("### 최종 저장\n위 셀에서 `submission.csv`(filename, airplane, bird, drone, index 없음)를 이미 저장했다. 아래 셀은 저장된 파일을 다시 읽어 확인한다.")] + [nbformat.v4.new_code_cell("df = pd.read_csv('submission.csv')\nprint(df.shape, list(df.columns))\ndf.head()")]
    for c in target.cells:
        if c.cell_type == "code": c.outputs = []; c.execution_count = None
# 초기 코드 셀 5, 6(import / yolov8s 로드)은 그대로 둔다. 로컬에서는 yolov8s.pt 를 같은 폴더에 둔 파일을 쓴다.
NotebookClient(work, timeout=3000, kernel_name="python3", resources={"metadata": {"path": os.getcwd()}}).execute()
for i in (1, 2): work.cells[i].source = nb.cells[i].source; work.cells[i].outputs = []; work.cells[i].execution_count = None     # 원문 복원
nbformat.write(work, "비행 물체 인식하기.ipynb"); print("saved")
