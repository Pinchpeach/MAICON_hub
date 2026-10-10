"""문제 2·4 발표자료 요약판(8쪽). build_deck.py 의 차트·표 도우미와 스타일을 그대로 쓴다."""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_deck as D
from build_deck import bars, gbars, hstack, tbl, BLUE, ORANGE

D.S.clear(); HERE = D.HERE
slide = D.slide

# 1 표지
slide("", "", """
<div style="position:absolute;left:0;top:0;width:1280px;height:720px;background:linear-gradient(135deg,#eaf2fc 0%,#fcfcfb 62%)"></div>
<div style="position:absolute;left:80px;top:170px;right:80px">
  <div class="pill">MAICON 모의고사 1회차 · 2026.10.09 · 요약</div>
  <div style="font-size:56px;font-weight:700;line-height:1.2;margin-top:22px">사진 속 인원·차량 집계<br><span style="color:#2a78d6">YOLOv8n 사전학습 추론 + 다중 관점 융합</span></div>
  <div style="font-size:24px;color:#52514e;margin-top:30px;line-height:1.55">문제 2 · 사람 수 &nbsp;|&nbsp; 문제 4 · 사람 수 + 차량 수<br>전처리 · 모델 · 성능지표 핵심만</div>
</div>""", "", cover=True)

# 2 결과 한눈에
slide("결과", "학습 없이 '여러 번 보고 합치기'로 템플릿 대비 점수 +19~31", f"""
<div class="grid g2" style="margin-bottom:18px">
  <div class="card"><h3>문제 2 · 사람 수</h3><div style="display:flex;gap:30px;align-items:baseline"><div><div class="big">57.7점</div><div class="small">템플릿 26.8점</div></div><div><div style="font-size:28px;font-weight:700">MAE 0.733</div><div class="small">템플릿 2.733</div></div></div></div>
  <div class="card"><h3>문제 4 · 사람 + 차량</h3><div style="display:flex;gap:30px;align-items:baseline"><div><div class="big">72.3점</div><div class="small">템플릿 53.6점</div></div><div><div style="font-size:28px;font-weight:700">MAE 0.383</div><div class="small">사람 0.333 · 차량 0.433</div></div></div></div>
</div>
<ul style="font-size:21px;line-height:1.7">
 <li><b>모델</b>: COCO 사전학습 <b>yolov8n</b> (학습하지 않고 추론만)</li>
 <li><b>핵심 아이디어</b>: 해상도·좌우 반전·타일로 여러 번 추론 → 박스를 한 객체 단위로 합침 → 점수 문턱 이상만 센다</li>
 <li><b>문제별로 다른 것</b>: 관점 구성, 임계값, (문제 2) 예측 하한 4, (차량) 면적비 규칙</li>
 <li><b>점수</b>: 100 / (1 + MAE), 공개 학습 30장 기준 추정 (평가 정답은 비공개)</li></ul>
""", "개선의 대부분은 모델을 바꾼 것이 아니라 관점을 늘리고 박스를 합치는 방식에서 나왔다.")

# 3 데이터와 전처리
slide("데이터 · 전처리", "작고 수량만 있는 데이터를 '낮은 문턱 + 정규화 좌표'로 받는다", f"""
<div class="grid g3" style="gap:12px;margin-bottom:12px">
 <div class="card" style="padding:8px 12px"><b style="color:#1c5cab">문제 2 · 사람 수</b>{bars(list(range(4,14)),[7,5,6,2,2,2,3,0,1,2],w=336,h=150,ymax=8,label_size=12)}<div class="tiny">전부 4명 이상 · 평균 6.83</div></div>
 <div class="card" style="padding:8px 12px"><b style="color:#1c5cab">문제 4 · 사람 수</b>{bars(list(range(0,12)),[10,7,2,4,3,0,0,1,1,1,0,1],w=336,h=150,ymax=12,label_size=12)}<div class="tiny">0명이 10장 · 평균 2.33</div></div>
 <div class="card" style="padding:8px 12px"><b style="color:#1c5cab">문제 4 · 차량 수</b>{bars(list(range(0,8)),[8,7,6,3,1,4,0,1],w=336,h=150,ymax=12,label_size=12)}<div class="tiny">0대 8장 · 평균 1.97</div></div>
</div>
<div class="grid g2" style="gap:14px;font-size:17px;line-height:1.5">
 <div class="card" style="padding:10px 16px"><h3 style="font-size:19px">데이터</h3><ul style="margin:2px 0"><li>학습 30장 / 평가 30장 (COCO 생활 사진), <b>수량 라벨만 있고 박스 라벨 없음</b></li><li>차량 = car + bus + truck (자전거·오토바이 제외)</li><li>수량 라벨은 임계값 선택과 검증에만 사용</li></ul></div>
 <div class="card" style="padding:10px 16px"><h3 style="font-size:19px">전처리</h3><ul style="margin:2px 0"><li><b>입력</b>: RGB→BGR, letterbox 640/960/1280, 깨진 파일은 0 처리</li><li><b>후보</b>: conf ≥ 0.03으로 넓게 받고, 박스는 원본 기준 0~1 좌표로 통일</li><li><b>관점</b>: 해상도 · 좌우 반전 · 2×2 타일(문제 2)</li><li><b>버린 것</b>: 밝기·대비 정규화 6종, 크기 변형 → 개선 없음</li></ul></div>
</div>
""", "분포가 달라 문제마다 관점과 문턱을 따로 정했고, 이미지 보정 전처리는 효과가 없었다.")

# 4 모델
slide("모델", "YOLOv8n 사전학습 추론 → 박스 융합 → 문턱으로 개수 결정", f"""
<div class="flow" style="margin-bottom:12px">
 <div class="step"><span class="no">1</span><b>관점 N개</b>해상도·반전·타일</div><div class="arrow">›</div>
 <div class="step"><span class="no">2</span><b>YOLOv8n</b>COCO 80클래스, conf ≥ 0.03</div><div class="arrow">›</div>
 <div class="step"><span class="no">3</span><b>그룹</b>사람 / 차량(car+bus+truck)</div><div class="arrow">›</div>
 <div class="step"><span class="no">4</span><b>융합</b>군집·투표·점수</div><div class="arrow">›</div>
 <div class="step"><span class="no">5</span><b>개수</b>점수 ≥ 임계값, 하한</div>
</div>
<div class="grid g2" style="gap:14px;align-items:start">
 <div class="card" style="padding:10px 16px;font-size:17px;line-height:1.5"><h3 style="font-size:19px">융합 규칙</h3><ul style="margin:2px 0"><li>점수 순으로 박스를 보며 <b>IoU ≥ 0.5 또는 포함도 0.7~0.8</b>이면 같은 객체(군집)</li><li>군집 점수 = <b>관점별 최고 conf의 합 ÷ 관점 수</b></li><li><b>투표</b>: 본 관점이 2개 이상인 군집만(문제 4 사람은 1)</li><li>차량: <b>면적비 &lt; 0.1</b>이면 큰 차 안의 작은 차를 따로 셈 (MAE 0.50 → 0.43)</li></ul></div>
 <div>{tbl(["", "문제 2 사람", "문제 4 사람", "문제 4 차량"], [
  ["관점", "960·1280·타일", "640·960·1280 ×(원본, 반전)", "640·960·1280"],
  ["관점 수", "3", "6", "3"], ["최소 관점", "2", "1", "2"], ["포함도", "0.7", "0.8", "0.7"], ["면적비", "-", "-", "0.1"],
  ["임계값", "0.15", "0.15", "0.175"], ["하한", "4", "0", "0"]], widths=["16%", "26%", "34%", "24%"], hl=(5, 6), tight=True)}</div>
</div>
""", "모델은 고정하고, 어떻게 보고(관점) 어떻게 합치는지(융합)로 성능을 만들었다.")

# 5 성능지표
slide("성능지표", "점수 = 100 / (1 + MAE): 템플릿 대비 MAE 절반 이하", f"""
<div class="grid g2" style="gap:14px;margin-bottom:10px">
 <div class="card" style="padding:10px 14px"><div style="font-weight:700">MAE (낮을수록 좋음)</div>{gbars(["문제 2 사람", "문제 4 사람", "문제 4 차량"], [[2.733, 0.933, 0.800], [0.733, 0.333, 0.433]], w=528, h=215, ymax=3.2, fmt="{:.2f}", names=("템플릿(416, conf 0.25)", "최종 메인"))}</div>
 <div class="card" style="padding:10px 16px;font-size:17px;line-height:1.5"><b style="color:#1c5cab">지표 읽는 법</b><ul style="margin:4px 0"><li><b>MAE</b> = 이미지별 |예측 − 정답|의 평균. 문제 4는 사람·차량 MAE의 평균</li><li><b>점수</b>: MAE 0.5 → 66.7, 1.0 → 50, 2.7 → 약 27</li><li><b>LOO</b>: 29장으로 임계값을 고르고 1장 평가(과대평가 점검)</li><li><b>편향</b>: 음수 = 덜 센다</li><li>30장이라 <b>1장 = MAE 0.033</b></li></ul></div>
</div>
{tbl(["", "MAE", "LOO MAE", "완전 일치", "편향", "점수 환산 (템플릿)"], [
 ["문제 2 사람", "0.733", "0.733", "0.533", "−0.533", "57.7 (26.8)"],
 ["문제 4 사람 / 차량", "0.333 / 0.433", "0.333 / 0.467", "0.700 / 0.600", "−0.067 / −0.300", "-"],
 ["문제 4 평균", "0.383", "0.400", "둘 다 0.367", "-", "72.3 (53.6), LOO 71.4"]], nums=(1, 2, 4), hl=(2,), widths=["19%", "16%", "16%", "16%", "15%", "18%"], tight=True)}
""", "두 문제 모두 '덜 세는' 음수 편향이 남아 있고, 문제 2와 차량에서 특히 크다.")

# 6 단계별 향상
slide("성능 향상 요인", "가장 큰 개선은 '단일 관점 → 다중 관점 융합'", f"""
<div class="grid g3" style="gap:12px">
<div class="card"><h3>문제 2 · 사람</h3>{bars(["템플릿", "단일 해상도|(1280)", "다중 관점|융합", "예측|하한 4"], [2.73, 1.20, 0.97, 0.73], w=334, h=225, ymax=3.2, fmt="{:.2f}", hl=3)}</div>
<div class="card"><h3>문제 4 · 사람</h3>{bars(["템플릿", "단일 해상도|(최선)", "다중 관점|+ 반전"], [0.93, 0.60, 0.33], w=334, h=225, ymax=1.2, fmt="{:.2f}", hl=2)}</div>
<div class="card"><h3>문제 4 · 차량</h3>{bars(["템플릿", "단일 해상도|(최선)", "다중 관점|융합", "면적비|규칙"], [0.80, 0.53, 0.50, 0.43], w=334, h=225, ymax=1.2, fmt="{:.2f}", hl=3)}</div>
</div>
<ul style="margin-top:12px;font-size:19px;line-height:1.6">
 <li>융합: 문제 4 사람 0.60 → 0.33 (−45%). 단일 해상도는 클수록 좋음(문제 2: 640/960/1280 = 1.60/1.40/1.20).</li>
 <li>임계값은 0.15~0.175 근처에서 MAE가 바닥인 U자 곡선이라 <b>이웃 평균이 가장 낮은 지점</b>으로 골랐다.</li>
 <li>예측 하한 4는 학습 3장에서 나온 개선이라 가장 가정 의존적이다.</li></ul>
<div class="tiny">주황 막대가 최종. 템플릿은 CPU, 단일 해상도는 GPU 실험 값.</div>
""", "모델 교체 없이 관점·융합·문턱만으로 MAE를 절반 이하로 줄였다.")

# 7 오류와 미채택
slide("오류 · 실험", "오류는 '못 보는 객체'에서 나와 후처리로는 더 줄이지 못했다", f"""
<div class="grid g2" style="gap:14px;align-items:start">
<div class="card" style="padding:10px 14px"><div style="font-weight:700;margin-bottom:2px">오답 이미지 수 (덜 셈 / 더 셈)</div>{hstack([("문제 2 사람", 11, 3), ("문제 4 사람", 5, 4), ("문제 4 차량", 10, 2)], w=520, row_h=38)}
<ul style="margin:6px 0 0;font-size:16px;line-height:1.45"><li><b>덜 센다</b>: 원경의 작은 사람, 가려진 사람, 유리 너머·야간·가장자리의 차</li><li><b>더 센다</b>: 마네킹·벽화 속 인물, 군중의 다리</li></ul>
<div class="tiny">문제 2는 GPU 캐시 기준(MAE 0.77).</div></div>
<div>{tbl(["시도 (채택 안 함)", "결과 (MAE)"], [
 ["박스를 직접 고쳐 yolov8n 미세조정", "문제 4 평균 0.45~0.55 (메인 0.383)"],
 ["보정 회귀 (개수 → 수량)", "문제 2 1.23~2.07 → 과적합"],
 ["관점 추가(1600, 3×3 타일)", "차이 없음"],
 ["임계값 아래 후보 확대 재검출", "문제 2 1.20 · 차량 0.40(3장 개선)"],
 ["1,035개 설정 전수 탐색", "최저 0.267/0.367 → 선택 노이즈"],
 ["평균 확신도 낮으면 +1 (문제 2)", "0.73 → 0.60, 평가 쪽 위험으로 미채택"]], widths=["52%", "48%"], tight=True)}
<div class="small" style="margin-top:8px">미세조정은 에폭을 늘릴수록 나빠졌다: 24장으로는 사전학습이 이미 아는 것을 흐트러뜨린다.</div></div>
</div>
""", "대부분의 '더 복잡한 방법'은 한 장씩 빼는 검증에서 나빠졌고 단순한 융합 규칙이 가장 안정적이었다.")

# 8 한계와 제출
slide("한계 · 제출", "주의할 점과 최종 제출", f"""
<div class="grid g2" style="gap:16px;align-items:start">
<div class="card o" style="padding:12px 18px"><h3>한계 · 위험</h3><ul style="font-size:17px;line-height:1.5;margin:2px 0">
 <li><b>검증 30장뿐</b>: 임계값·규칙을 같은 30장으로 골랐다(LOO 점검, 그래도 낙관적). 1장 = MAE 0.033.</li>
 <li><b>문제 2 하한 4</b>: 학습 정답이 4~13명이라는 가정. 평가에 4명 미만 이미지가 있으면 손해.</li>
 <li><b>분포</b>(문제 2 평균): 학습 정답 6.83 / 학습 예측 6.30 / 평가 예측 7.40.</li>
 <li><b>서버</b>: <code>ultralytics</code> 미설치(torch 1.10), 2코어 실행 시간 미측정.</li></ul></div>
<div class="card" style="padding:12px 18px"><h3>최종 제출</h3>{tbl(["", "문제 2", "문제 4"], [
 ["컬럼", "<code>person_count</code>", "<code>person_count</code>, <code>vehicle_count</code>"],
 ["행 수", "30 (ID 순서 일치)", "30 (ID 순서 일치)"],
 ["예측 합(평균)", "사람 222 (7.40)", "사람 67 (2.23) · 차량 50 (1.67)"],
 ["학습 점수 환산", "57.7", "72.3 (LOO 71.4)"]], widths=["26%", "30%", "44%"], tight=True)}
<div class="small" style="margin-top:8px"><code>work.ipynb</code> 처음부터 실행해 출력 저장, 오류 없음, CPU 약 33~41초(개발 PC).</div></div>
</div>
""", "사전학습 YOLOv8n을 여러 관점으로 보고 합치는 규칙만으로 점수를 +31(문제 2)·+19(문제 4) 올렸다.")

if __name__ == "__main__":
    p = os.path.join(HERE, "deck_brief.html"); open(p, "w", encoding="utf-8").write(D.render()); print("slides", len(D.S))
    if "--pdf" in sys.argv:
        edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"; out = os.path.join(HERE, "문제2_4_사진속인원차량집계_발표자료_요약.pdf")
        if os.path.exists(out): os.remove(out)
        subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}", "file:///" + p.replace("\\", "/").replace(" ", "%20")], capture_output=True)
        print("pdf", os.path.exists(out), os.path.getsize(out) // 1024, "KB")
