"""문제 2·4 (사진 속 인원·차량 집계) 발표 자료 생성. HTML 슬라이드 -> Edge(헤드리스) -> PDF. 숫자는 각 문제의 versions/EXPERIMENTS.md 와 노트북 출력에서 가져옴."""
import os, sys, subprocess, html

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, DEEP, ORANGE, AQUA = "#2a78d6", "#1c5cab", "#eb6834", "#1baf7a"
INK, INK2, MUTED, HAIR, WASH, SURF, GRAYBAR = "#0b0b0b", "#52514e", "#898780", "#e1e0d9", "#eaf2fc", "#fcfcfb", "#b9b8b2"
TITLE_FOOT = "문제 2·4 사진 속 인원·차량 집계"
DATE = "26.10.09"
CSS = """
*{box-sizing:border-box}html,body{margin:0;padding:0;background:#fff}
@page{size:1280px 720px;margin:0}
.slide{position:relative;width:1280px;height:720px;overflow:hidden;background:#fcfcfb;color:#0b0b0b;font-family:'Malgun Gothic','맑은 고딕',sans-serif;page-break-after:always;break-after:page}
.slide:last-child{page-break-after:auto;break-after:auto}
.kick{position:absolute;left:64px;top:34px;font-size:16px;font-weight:700;color:#2a78d6;letter-spacing:.02em}
h1{position:absolute;left:64px;top:58px;right:64px;margin:0;font-size:34px;line-height:1.25;font-weight:700}
.body{position:absolute;left:64px;right:64px;top:128px;bottom:126px;font-size:19px;line-height:1.5;color:#0b0b0b}
.take{position:absolute;left:64px;right:64px;bottom:50px;padding:11px 18px;background:#eaf2fc;border-left:5px solid #2a78d6;border-radius:4px;font-size:19px;line-height:1.4}
.take b{color:#1c5cab;margin-right:10px}
.foot{position:absolute;left:64px;right:64px;bottom:14px;border-top:1px solid #e1e0d9;padding-top:7px;font-size:13px;color:#898780;display:flex;justify-content:space-between}
.grid{display:grid;gap:22px}.g2{grid-template-columns:1fr 1fr}.g3{grid-template-columns:1fr 1fr 1fr}.g12{grid-template-columns:1.1fr 1fr}.g21{grid-template-columns:1fr 1.1fr}
.card{border:1px solid #e1e0d9;border-radius:10px;padding:14px 18px;background:#fff}
.card h3{margin:0 0 6px;font-size:20px;color:#1c5cab}
.card.o h3{color:#c4491c}.card.a h3{color:#0d7a54}
.card p,.card li{margin:3px 0}
ul{margin:4px 0;padding-left:22px}li{margin:3px 0}
table{border-collapse:collapse;width:100%;font-size:17px}
th{background:#eaf2fc;color:#1c5cab;text-align:left;padding:7px 10px;font-weight:700;border-bottom:2px solid #2a78d6}
td{padding:6px 10px;border-bottom:1px solid #e1e0d9;vertical-align:top}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr.hl td{background:#eaf2fc;font-weight:700}
table.tight{font-size:16px}table.tight td{padding:3px 10px}table.tight th{padding:5px 10px}
.big{font-size:44px;font-weight:700;color:#2a78d6;line-height:1.1}
.small{font-size:15px;color:#52514e}.muted{color:#898780}.tiny{font-size:13px;color:#898780}
.pill{display:inline-block;padding:2px 10px;border-radius:12px;background:#eaf2fc;color:#1c5cab;font-size:15px;font-weight:700;margin-right:6px}
.pill.o{background:#fdeee6;color:#c4491c}.pill.g{background:#eeeeea;color:#52514e}
.flow{display:flex;align-items:stretch;gap:0}
.step{flex:1;border:1px solid #e1e0d9;border-radius:8px;padding:10px 10px 8px;background:#fff;font-size:16px;line-height:1.35;position:relative}
.step b{display:block;color:#1c5cab;margin-bottom:3px;font-size:17px}
.step .no{position:absolute;top:-11px;left:10px;background:#2a78d6;color:#fff;border-radius:11px;width:22px;height:22px;text-align:center;font-size:14px;line-height:22px;font-weight:700}
.arrow{width:26px;display:flex;align-items:center;justify-content:center;color:#898780;font-size:22px}
code{font-family:Consolas,'Malgun Gothic',monospace;background:#f1f0ec;padding:1px 6px;border-radius:4px;font-size:16px}
.fig{display:block;border:1px solid #e1e0d9;border-radius:8px;width:100%}
svg text{font-family:'Malgun Gothic','맑은 고딕',sans-serif}
"""

# ---------------- SVG 차트 도우미 ----------------
def esc(s): return html.escape(str(s))

def bars(cats, vals, w=380, h=210, color=BLUE, ymax=None, fmt="{:g}", ytitle="", hide0=True, label_size=13, hl=None):
    """세로 막대. hl: 강조할 인덱스(다른 색)"""
    l, r, t, b = 34, 8, 14, 42 if any('|' in str(c) for c in cats) else 28; ymax = ymax or max(vals) * 1.18 or 1; pw, ph = w - l - r, h - t - b; n = len(cats); bw = pw / n * 0.62; o = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    for k in range(0, 5):
        y = t + ph - ph * k / 4; v = ymax * k / 4; o.append(f'<line x1="{l}" x2="{w - r}" y1="{y:.1f}" y2="{y:.1f}" stroke="{HAIR}" stroke-width="1"/>'); o.append(f'<text x="{l - 5}" y="{y + 4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">{fmt.format(round(v, 2)) if v < 10 else int(round(v))}</text>')
    for i, (c, v) in enumerate(zip(cats, vals)):
        x = l + pw / n * (i + 0.5) - bw / 2; bh = ph * v / ymax; y = t + ph - bh; col = ORANGE if (hl is not None and i == hl) else color
        if v > 0: o.append(f'<path d="M{x:.1f},{t + ph:.1f} V{y + 3:.1f} Q{x:.1f},{y:.1f} {x + 3:.1f},{y:.1f} H{x + bw - 3:.1f} Q{x + bw:.1f},{y:.1f} {x + bw:.1f},{y + 3:.1f} V{t + ph:.1f} Z" fill="{col}"/>')
        if v > 0 or not hide0: o.append(f'<text x="{x + bw / 2:.1f}" y="{y - 4:.1f}" font-size="{label_size}" font-weight="700" fill="{INK}" text-anchor="middle">{fmt.format(v)}</text>')
        parts = str(c).split('|'); o.append(f'<text x="{x + bw / 2:.1f}" y="{h - 10 - 13 * (len(parts) - 1)}" font-size="12" fill="{INK2}" text-anchor="middle">' + "".join(f'<tspan x="{x + bw / 2:.1f}" dy="{0 if k == 0 else 13}">{esc(pp)}</tspan>' for k, pp in enumerate(parts)) + '</text>')
    o.append(f'<line x1="{l}" x2="{w - r}" y1="{t + ph}" y2="{t + ph}" stroke="{INK2}" stroke-width="1"/></svg>'); return "".join(o)

def gbars(cats, series, w=520, h=250, ymax=None, fmt="{:g}", colors=(GRAYBAR, BLUE), names=("", "")):
    """묶음 막대 (카테고리마다 막대 2개). series: [[v..],[v..]]"""
    l, r, t, b = 36, 8, 28, 30; ymax = ymax or max(max(s) for s in series) * 1.15; pw, ph = w - l - r, h - t - b; n = len(cats); gw = pw / n; bw = gw * 0.30; o = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    for k in range(0, 5):
        y = t + ph - ph * k / 4; v = ymax * k / 4; o.append(f'<line x1="{l}" x2="{w - r}" y1="{y:.1f}" y2="{y:.1f}" stroke="{HAIR}"/><text x="{l - 5}" y="{y + 4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">{int(round(v)) if ymax > 10 else round(v, 1)}</text>')
    for i, c in enumerate(cats):
        for j, s in enumerate(series):
            v = s[i]; x = l + gw * i + gw * 0.5 + (j - 1) * (bw + 4) + 2; bh = ph * v / ymax; y = t + ph - bh
            o.append(f'<path d="M{x:.1f},{t + ph:.1f} V{y + 3:.1f} Q{x:.1f},{y:.1f} {x + 3:.1f},{y:.1f} H{x + bw - 3:.1f} Q{x + bw:.1f},{y:.1f} {x + bw:.1f},{y + 3:.1f} V{t + ph:.1f} Z" fill="{colors[j]}"/><text x="{x + bw / 2:.1f}" y="{y - 4:.1f}" font-size="13" font-weight="700" fill="{INK}" text-anchor="middle">{fmt.format(v)}</text>')
        o.append(f'<text x="{l + gw * (i + 0.5):.1f}" y="{h - 9}" font-size="13" fill="{INK2}" text-anchor="middle">{esc(c)}</text>')
    lx = l
    for j, nm in enumerate(names):
        o.append(f'<rect x="{lx}" y="5" width="12" height="12" rx="2" fill="{colors[j]}"/><text x="{lx + 17}" y="16" font-size="13" fill="{INK2}">{esc(nm)}</text>'); lx += 17 + 13 * len(nm) + 18
    o.append(f'<line x1="{l}" x2="{w - r}" y1="{t + ph}" y2="{t + ph}" stroke="{INK2}"/></svg>'); return "".join(o)

def lines(xs, ys_list, names, colors, marks, w=640, h=300, ymax=1.8):
    """꺾은선 + 선택 지점 표시. marks: 시리즈별 (x인덱스)"""
    l, r, t, b = 44, 150, 16, 34; pw, ph = w - l - r, h - t - b; X = lambda i: l + pw * i / (len(xs) - 1); Y = lambda v: t + ph - ph * v / ymax; o = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    for k in range(0, 5):
        v = ymax * k / 4; o.append(f'<line x1="{l}" x2="{l + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{HAIR}"/><text x="{l - 6}" y="{Y(v) + 4:.1f}" font-size="12" fill="{MUTED}" text-anchor="end">{v:.1f}</text>')
    for i, x in enumerate(xs):
        if i % 2 == 1 or i == 0: o.append(f'<text x="{X(i):.1f}" y="{h - 12}" font-size="12" fill="{INK2}" text-anchor="middle">{x}</text>')
    for ys, nm, col, mk in zip(ys_list, names, colors, marks):
        o.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.5" stroke-linejoin="round" points="{" ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(ys))}"/>')
        for i, v in enumerate(ys): o.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3" fill="{col}" stroke="{SURF}" stroke-width="1.5"/>')
        o.append(f'<circle cx="{X(mk):.1f}" cy="{Y(ys[mk]):.1f}" r="7.5" fill="none" stroke="{INK}" stroke-width="2"/><text x="{X(mk):.1f}" y="{Y(ys[mk]) + 24:.1f}" font-size="12" font-weight="700" fill="{INK}" text-anchor="middle">{ys[mk]:.2f}</text>')
        o.append(f'<rect x="{l + pw + 8}" y="{Y(ys[-1]) - 2:.1f}" width="14" height="4" rx="2" fill="{col}"/><text x="{l + pw + 27}" y="{Y(ys[-1]) + 4:.1f}" font-size="13" font-weight="700" fill="{INK}">{esc(nm)}</text>')
    o.append(f'<line x1="{l}" x2="{l + pw}" y1="{t + ph}" y2="{t + ph}" stroke="{INK2}"/><text x="{l + pw / 2:.1f}" y="{h - 0}" font-size="12" fill="{MUTED}" text-anchor="middle"></text></svg>'); return "".join(o)

def hstack(rows, w=520, row_h=44, colors=(BLUE, ORANGE), names=("덜 셈(과소)", "더 셈(과다)"), unit="장"):
    """가로 누적 막대. rows: [(라벨, a, b)]"""
    l = 150; mx = max(a + b for _, a, b in rows); scale = (w - l - 100) / mx; h = row_h * len(rows) + 34; o = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    lx = l
    for j, nm in enumerate(names): o.append(f'<rect x="{lx}" y="4" width="12" height="12" rx="2" fill="{colors[j]}"/><text x="{lx + 17}" y="15" font-size="13" fill="{INK2}">{esc(nm)}</text>'); lx += 17 + 13 * len(nm) + 16
    for i, (nm, a, b) in enumerate(rows):
        y = 30 + i * row_h; o.append(f'<text x="{l - 10}" y="{y + 22}" font-size="15" fill="{INK}" text-anchor="end">{esc(nm)}</text>'); x = l
        for j, v in enumerate((a, b)):
            if v: o.append(f'<rect x="{x:.1f}" y="{y}" width="{v * scale - 2:.1f}" height="30" rx="3" fill="{colors[j]}"/><text x="{x + v * scale / 2 - 1:.1f}" y="{y + 21}" font-size="15" font-weight="700" fill="#fff" text-anchor="middle">{v}{unit}</text>'); x += v * scale
        o.append(f'<text x="{x + 8:.1f}" y="{y + 21}" font-size="14" fill="{INK2}">합계 {a + b}</text>')
    o.append("</svg>"); return "".join(o)

# ---------------- 슬라이드 ----------------
S = []
def slide(kick, title, body, take, cover=False):
    S.append(dict(kick=kick, title=title, body=body, take=take, cover=cover))

def tbl(head, rows, nums=(), hl=(), widths=None, tight=False):
    h = "".join(f'<th class="{"n" if i in nums else ""}"' + (f' style="width:{widths[i]}"' if widths else "") + f'>{c}</th>' for i, c in enumerate(head))
    b = "".join(f'<tr class="{"hl" if k in hl else ""}">' + "".join(f'<td class="{"n" if i in nums else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for k, r in enumerate(rows))
    return f'<table class="{"tight" if tight else ""}"><tr>{h}</tr>{b}</table>'

# 1 표지
slide("", "", f"""
<div style="position:absolute;left:0;top:0;width:1280px;height:720px;background:linear-gradient(135deg,#eaf2fc 0%,#fcfcfb 62%)"></div>
<div style="position:absolute;left:80px;top:150px;right:80px">
  <div class="pill">MAICON 모의고사 1회차 · 2026.10.09</div>
  <div style="font-size:56px;font-weight:700;line-height:1.2;margin-top:22px">사진 속 인원·차량 집계<br><span style="color:#2a78d6">YOLOv8n 사전학습 추론 + 다중 관점 융합</span></div>
  <div style="font-size:24px;color:#52514e;margin-top:30px;line-height:1.55">문제 2 · 사진 속 인원 집계 &nbsp;|&nbsp; 문제 4 · 사진 속 인원·차량 집계<br>데이터 전처리 · 모델 · 성능지표 중심으로 정리</div>
  <div style="margin-top:46px;font-size:18px;color:#898780">COCO 2017 생활 사진 학습 30장 · 평가 30장 (문제당) &nbsp;·&nbsp; 지표: 100 / (1 + Count MAE)</div>
</div>""", "", cover=True)

# 2 한눈에
slide("개요", "두 문제의 목표와 최종 결과", f"""
<div class="grid g2" style="margin-bottom:16px">
  <div class="card"><h3>문제 2 · 사진 속 인원 (S11-2)</h3><div class="small">예측: 사람 수 <code>person_count</code></div>
    <div style="display:flex;gap:26px;align-items:baseline;margin-top:6px"><div><div class="big">57.7</div><div class="small">학습 30장 환산 점수</div></div><div><div style="font-size:28px;font-weight:700">MAE 0.733</div><div class="small">템플릿 2.733 (26.8점)</div></div></div></div>
  <div class="card"><h3>문제 4 · 사진 속 인원·차량 (S12-2)</h3><div class="small">예측: 사람 수 + 차량 수(승용차·버스·트럭)</div>
    <div style="display:flex;gap:26px;align-items:baseline;margin-top:6px"><div><div class="big">72.3</div><div class="small">학습 30장 환산 점수</div></div><div><div style="font-size:28px;font-weight:700">MAE 0.383</div><div class="small">템플릿 0.867 (53.6점)</div></div></div></div>
</div>
{tbl(["", "문제 2", "문제 4"], [
 ["공통 풀이", "COCO 사전학습 <b>yolov8n</b> 추론만 (학습 없음) → 여러 관점의 박스를 합쳐 중복 제거 → 임계값 이상만 센다", "문제 2와 같은 방식, 사람·차량을 따로 설정"],
 ["서로 다른 점", "사람이 많고 작다(정답 4~13명): 큰 해상도 + 타일, 예측 하한 4", "사람 적고(0~11) 차량 0~7대: 반전 포함 6관점, 차량은 크기 비율 규칙"],
 ["제출 파일", "<code>filename,person_count</code> 30행", "<code>filename,person_count,vehicle_count</code> 30행"]], widths=["16%", "44%", "40%"])}
""", "공통 구조는 같고, 사람이 많고 작은 장면(문제 2)과 적고 큰 장면(문제 4)에서 최적 설정이 달랐다.")

# 3 문제와 데이터
slide("데이터", "문제 정의와 제공 데이터", f"""
{tbl(["항목", "문제 2", "문제 4"], [
 ["입력", "COCO 2017 validation 생활 사진 (군 행사 사진 아님)", "COCO 2017 validation 생활 사진"],
 ["학습 / 평가", "30장 / 30장 (정답 수량 <code>person_count</code>만 있고 <b>박스 라벨 없음</b>)", "30장 / 30장 (<code>person_count</code>, <code>vehicle_count</code>, 박스 라벨 없음)"],
 ["정답 정의", "COCO <code>person</code> 객체 수, <code>iscrowd=1</code> 이미지는 제외", "사람 = COCO <code>person</code>, 차량 = <code>car + bus + truck</code> (자전거·오토바이 제외)"],
 ["학습 정답 요약", "합 205명, 평균 6.83, 범위 4~13", "사람 합 70 (평균 2.33, 0~11) · 차량 합 59 (평균 1.97, 0~7)"],
 ["이미지 크기", "640×480 8장, 640×427 6장, 640×426 2장 등 (긴 변 640)", "640×480 6장, 640×427 4장, 480×640 3장 등"],
 ["보조 파일", "<code>train_labels.csv</code>에 <code>unit_name</code>(가상 부대), <code>event_date</code>가 함께 있음", "<code>image_credits.csv</code>(이미지 출처·라이선스)"],
 ["평가", "점수 = 100 / (1 + Count MAE), 보조: 완전 일치율", "점수 = 100 / (1 + <b>클래스 평균</b> MAE)"]], widths=["15%", "42%", "43%"])}
<div class="small" style="margin-top:10px">두 문제는 서로 다른 60장씩이며 정답은 모델 추론값이 아닌 공개 객체 주석에서 만든 값이다. 평가용 정답을 원본 주석에서 조회하는 것은 금지(<code>image_credits.csv</code>의 COCO 이미지 번호는 사용하지 않음).</div>
""", "정답은 '수량'뿐이라 박스 학습이 불가능하고, 수량 라벨은 임계값 선택과 검증에만 쓴다.")

# 4 정답 분포
slide("데이터 분석", "정답 수량 분포: 문제 2는 '많은 사람', 문제 4는 '적은 사람'", f"""
<div class="grid g3" style="gap:14px">
 <div class="card"><h3>문제 2 · 사람 수</h3>{bars([4,5,6,7,8,9,10,11,12,13],[7,5,6,2,2,2,3,0,1,2],w=334,h=215,ymax=8)}<div class="small">전부 4명 이상 · 평균 6.83 · 표준편차 2.72</div></div>
 <div class="card"><h3>문제 4 · 사람 수</h3>{bars(list(range(0,12)),[10,7,2,4,3,0,0,1,1,1,0,1],w=334,h=215,ymax=12)}<div class="small">0명이 10장(33%) · 평균 2.33 · 표준편차 2.89</div></div>
 <div class="card"><h3>문제 4 · 차량 수</h3>{bars(list(range(0,8)),[8,7,6,3,1,4,0,1],w=334,h=215,ymax=12)}<div class="small">0대 8장 · 5대 이상 5장 · 평균 1.97</div></div>
</div>
<ul style="margin-top:12px;font-size:18px">
 <li>세로축은 이미지 장수. 문제 2는 <b>4명 미만 이미지가 하나도 없어서</b> 예측 하한 4를 쓸 근거가 생겼다(위험은 마지막 쪽에).</li>
 <li>문제 4는 정답 0이 흔해서 <b>오탐(없는 것을 세는 것)</b>도 점수를 깎는다 → 임계값을 너무 낮추면 안 된다.</li>
 <li>수량이 정수이고 이미지당 최대 13이라, 1장의 1명 오차가 MAE 0.033(=1/30)에 해당한다.</li>
</ul>""", "분포가 달라서 같은 임계값·같은 해상도를 두 문제에 그대로 쓸 수 없다.")

# 5 전처리 ① 입력 처리
slide("데이터 전처리 ①", "입력 처리: 이미지 → 모델이 보는 배열 → 통일된 박스 좌표", f"""
<div class="flow" style="margin-bottom:16px">
 <div class="step"><span class="no">1</span><b>읽기</b>PIL로 열어 RGB 변환 후 BGR 배열(ultralytics 규약). 깨진 파일은 '빈 결과'(0명)로 대체</div><div class="arrow">›</div>
 <div class="step"><span class="no">2</span><b>크기 맞춤</b>비율 유지 letterbox, 정사각형 입력 <code>imgsz</code> 640 / 960 / 1280</div><div class="arrow">›</div>
 <div class="step"><span class="no">3</span><b>후보 박스</b>conf ≥ 0.03, NMS IoU 0.7, 최대 300개 (낮은 문턱으로 넓게 받고 나중에 거른다)</div><div class="arrow">›</div>
 <div class="step"><span class="no">4</span><b>좌표 통일</b>모든 박스를 원본 이미지 기준 0~1 정규화 xyxy로 변환 (반전·타일은 되돌림)</div><div class="arrow">›</div>
 <div class="step"><span class="no">5</span><b>클래스 묶기</b>사람 = person(0), 차량 = car(2)+bus(5)+truck(7)</div>
</div>
<div class="grid g2">
 <div class="card"><h3>왜 이렇게 했나</h3><ul>
  <li><b>낮은 문턱(0.03)</b>: 먼 곳의 작은 사람은 점수가 0.04~0.25로 낮아서 처음부터 높게 거르면 사라진다.</li>
  <li><b>정규화 좌표</b>: 해상도·타일·반전이 다른 관점의 박스를 한 좌표계에서 겹침 비교하려는 목적.</li>
  <li><b>차량 한 묶음</b>: 같은 차가 car·truck 두 클래스로 잡혀도 한 대로 센다.</li></ul></div>
 <div class="card o"><h3>하지 않은 것</h3><ul>
  <li>사람과 차량 사이의 중복 제거 — 버스 안의 사람처럼 실제로 겹치는 다른 객체가 지워진다.</li>
  <li>이미지 밝기·대비 정규화 — 시험했으나 개선 없음(7쪽).</li>
  <li>수량 라벨로 박스를 학습 — 박스가 없고 30장뿐(16쪽 실험에서 오히려 악화).</li></ul></div>
</div>""", "낮은 문턱으로 후보를 넓게 받고, 정규화 좌표에서 합치고, 개수는 마지막 임계값으로 정한다.")

# 6 전처리 ② 관점
slide("데이터 전처리 ②", "관점(view) 만들기: 해상도 · 좌우 반전 · 2×2 타일", f"""
<div class="grid g12" style="align-items:start">
<div>{tbl(["관점", "만드는 법", "노리는 것"], [
 ["해상도 640 / 960 / 1280", "같은 이미지를 3가지 크기로 추론", "큰 해상도일수록 작은 사람이 잡힘 (문제 2 단일 MAE 1.60 → 1.40 → 1.20)"],
 ["좌우 반전", "이미지를 뒤집어 추론한 뒤 박스 좌표를 되돌림", "같은 객체를 다시 보고 점수 보강"],
 ["2×2 타일(문제 2)", "이미지의 62.5% 크기 조각 4개(25% 겹침)를 각각 추론", "원경의 아주 작은 사람"]], widths=["24%", "36%", "40%"])}
<div class="small" style="margin-top:10px">타일은 조각별 개수를 더하지 않고, 박스를 원본 좌표로 되돌려 한 군집 규칙으로 합친 뒤 센다(경계에 걸친 사람의 중복 방지).</div></div>
<div>{tbl(["", "관점 구성", "수"], [
 ["문제 2 사람", "960, 1280, 타일(960)", "3"],
 ["문제 4 사람", "640·960·1280 × (원본, 반전)", "6"],
 ["문제 4 차량", "640·960·1280 (반전 없음)", "3"]], nums=(2,), widths=["26%", "58%", "16%"])}
<div class="card" style="margin-top:12px"><b>문제마다 관점이 다른 이유</b><br><span class="small">서로의 규칙을 바꿔 적용하면 나빠졌다: 문제 2 규칙 → 문제 4 사람 MAE 0.77 (자기 규칙 0.33), 문제 4 규칙 → 문제 2 1.47 (자기 규칙 0.77). 사람이 많고 작은 장면과 적고 큰 장면은 최적 규칙이 다르다.</span></div>
</div></div>
<div class="small" style="margin-top:8px">사용하지 않은 열: <code>unit_name</code>(가상 부대 이름)·<code>event_date</code> — 이미지만으로 푼다. 부대별 평균이 5.5~7.9로 보이지만 부대당 7~8장이라 근거로 삼기 어렵다(검증하지 않음).</div>
""", "관점은 '무엇을 놓치는가'에 맞춰 골랐고, 문제 간 규칙을 서로 옮기면 오히려 나빠진다.")

# 7 전처리 ③ 시험하고 버린 전처리
slide("데이터 전처리 ③", "밝기·대비 정규화와 크기 변형: 모두 개선 없음", f"""
<div class="grid g2" style="gap:18px">
<div><div style="font-weight:700;margin-bottom:6px">문제 2 사람 · MAE (한 장씩 빼는 검증 LOO)</div>{tbl(["전처리", "MAE", "LOO", "판정"], [
 ["원본 (기준)", "0.97", "0.97", "유지"], ["자동 대비 1~99%", "1.00", "1.00", "기각"], ["감마(평균 0.45)", "1.03", "1.23", "기각"], ["CLAHE", "1.33", "1.70", "기각"], ["히스토그램 평활화", "0.87", "1.13", "기각*"], ["선명화", "1.03", "1.23", "기각"], ["밝기별 임계값 2개", "-", "1.03", "기각"]], nums=(1, 2), hl=(0,), widths=["44%", "17%", "17%", "22%"])}
<div class="tiny" style="margin-top:4px">* 한 장씩 빼는 검증에서 나빠져 기각(같은 학습 30장으로만 좋아 보임). 하한 4 적용 전 기준.</div></div>
<div><div style="font-weight:700;margin-bottom:6px">문제 4 · 사람 MAE (LOO) / 차량 MAE (LOO)</div>{tbl(["전처리", "사람", "차량", "판정"], [
 ["원본 (기준)", "0.33 (0.33)", "0.50 (0.53)", "유지"], ["자동 대비", "0.37 (0.47)", "0.53 (0.57)", "기각"], ["감마", "0.33 (0.33)", "0.67 (0.73)", "기각"], ["CLAHE", "0.40 (0.43)", "0.50 (0.53)", "기각"], ["히스토그램 평활화", "0.33 (0.40)", "0.67 (0.73)", "기각"], ["선명화", "0.47 (0.50)", "0.60 (0.57)", "기각"], ["밝기별 임계값 2개", "(0.47)", "(0.57)", "기각"]], nums=(1, 2), hl=(0,), widths=["38%", "24%", "24%", "14%"])}
<div class="tiny" style="margin-top:4px">차량 기준은 면적비 규칙(v3) 적용 전 값(0.50).</div></div>
</div>
<ul style="margin-top:12px;font-size:17px">
 <li>오류와 밝기의 상관(스피어만, 30장): 문제 2 −0.09 · 문제 4 사람 −0.13, 차량 +0.12 → 밝기와 오류는 관련이 약하다. 어두운 사진이 조금 나쁘지만 정규화로 줄지 않았다.</li>
 <li>크기 변형(문제 2, 하한 4 적용): 원본 0.77 / 0.75배 축소 1.37 / 0.5배 축소 1.67 / 1.5배 확대 0.87 — 사람이 작아질수록 덜 센다는 확인. 변형본을 섞어 임계값을 다시 골라도 0.15 그대로였다.</li>
</ul>""", "전처리로 이미지를 고치는 것보다, 놓치는 작은 객체를 더 보게 하는 관점 추가가 효과적이었다.")

# 8 모델 ① YOLOv8n + 흐름
slide("모델 ①", "YOLOv8n 사전학습 추론 + 후처리 파이프라인", f"""
<div class="flow" style="margin-bottom:16px">
 <div class="step"><span class="no">1</span><b>이미지</b>RGB 변환, 깨진 파일 0 처리</div><div class="arrow">›</div>
 <div class="step"><span class="no">2</span><b>관점 N개</b>해상도·반전·타일</div><div class="arrow">›</div>
 <div class="step"><span class="no">3</span><b>YOLOv8n</b>COCO 80클래스, conf ≥ 0.03</div><div class="arrow">›</div>
 <div class="step"><span class="no">4</span><b>그룹 선택</b>사람 / 차량</div><div class="arrow">›</div>
 <div class="step"><span class="no">5</span><b>박스 융합</b>군집·투표·점수</div><div class="arrow">›</div>
 <div class="step"><span class="no">6</span><b>임계값·하한</b>개수 결정</div>
</div>
<div class="grid g2">
 <div class="card"><h3>모델: yolov8n.pt (제공 가중치)</h3><ul>
  <li>COCO 80개 클래스로 사전학습된 객체 탐지 모델. 문제 폴더에 제공된 가중치 파일(6.5MB)을 그대로 사용 — 별도 다운로드 없음.</li>
  <li><b>학습하지 않고 추론만</b> 한다. COCO 클래스 중 <code>person</code>, <code>car</code>, <code>bus</code>, <code>truck</code>만 사용.</li>
  <li>CPU에서도 노트북 전체 약 33~41초(개발 PC 기준, 서버 2코어는 미측정).</li></ul></div>
 <div class="card o"><h3>학습하지 않은 이유</h3><ul>
  <li>박스 라벨이 없고 수량만 있다. 이미지는 학습 30장뿐.</li>
  <li>모델 예측을 박스 라벨로 만들어 미세조정한 실험(20에폭, 5겹 교차 검증)은 모두 사전학습 그대로보다 <b>나빴다</b>(16쪽).</li>
  <li>그래서 모델은 고정하고, <b>관점·융합·임계값</b>을 학습 30장으로 정했다.</li></ul></div>
</div>""", "사전학습 모델은 고정하고, 어떻게 보고(관점) 어떻게 합치는지(융합)가 성능을 좌우했다.")

# 9 모델 ② 융합 규칙
slide("모델 ②", "박스 융합 규칙: 여러 관점의 박스를 '한 객체 = 한 군집'으로", f"""
<div class="grid g2" style="gap:18px">
<div class="card"><h3>군집 만들기</h3><ol style="margin:4px 0;padding-left:22px;line-height:1.45">
 <li>모든 관점의 박스를 점수 높은 순으로 정렬한다.</li>
 <li>앞에서부터 보며, 이미 만든 군집의 <b>대표 박스</b>와 <code>IoU ≥ 0.5</code> 이거나, <b>작은 박스가 대표 안에 포함된 비율 ≥ 포함도</b>(0.7~0.8)이면 같은 객체로 합친다.</li>
 <li>아니면 새 군집(새 객체)을 만든다.</li>
 <li>차량은 <b>면적비(작은/큰) &lt; 0.1</b>이면 포함 관계여도 다른 객체로 둔다.</li></ol></div>
<div class="card"><h3>군집 점수 · 투표 · 개수</h3>
 <div style="font-size:20px;background:#f6f5f1;border-radius:6px;padding:8px 12px;margin:2px 0 8px">군집 점수 = Σ<sub>관점</sub> (그 관점의 최고 conf) ÷ 전체 관점 수</div>
 <ul><li><b>투표</b>: 그 객체를 본 관점이 vote 개 이상인 군집만 인정(문제 2·4 차량 2, 문제 4 사람 1).</li>
 <li><b>개수</b> = 점수 ≥ 임계값인 군집 수, 그 뒤 <b>예측 하한</b> 적용(문제 2는 4).</li>
 <li>여러 관점에서 일관되게 보인 객체일수록 점수가 높고, 한 관점에서만 잡힌 흐린 박스는 점수가 낮아져 걸러진다.</li></ul></div>
</div>
<div class="card o" style="margin-top:14px"><h3>왜 면적비 조건을 넣었나 (차량 MAE 0.50 → 0.43)</h3>
 포함도 0.7 규칙이 <b>큰 차량 박스 안에 보이는 뒤쪽의 작은 차</b>까지 같은 차로 합쳐 버렸다. 예: <code>S12_2_0010</code>(주황색 픽업트럭 뒤의 차 2대) 3대 → 5대로 정답(5)과 일치. 사람에는 같은 조건을 넣어도 변화가 없어(0.33 → 0.33) 적용하지 않았다.</div>
""", "군집 점수는 '몇 개 관점이 얼마나 확신했는가'이고, 개수는 그 점수의 문턱으로 정한다.")

# 10 모델 ③ 최종 설정
slide("모델 ③", "최종 설정 한눈에 보기", f"""
{tbl(["설정 항목", "문제 2 · 사람", "문제 4 · 사람", "문제 4 · 차량"], [
 ["사용 클래스", "person", "person", "car + bus + truck (한 묶음)"],
 ["관점", "full960 · full1280 · tile960", "640·960·1280 × (원본, 반전)", "640·960·1280"],
 ["관점 수", "3", "6", "3"],
 ["군집 IoU", "0.5", "0.5", "0.5"],
 ["포함도", "0.7", "0.8", "0.7"],
 ["최소 관점 수(vote)", "2", "1", "2"],
 ["면적비 조건", "없음", "없음", "0.1"],
 ["점수 임계값", "0.15", "0.15", "0.175"],
 ["예측 하한", "4 (학습 정답 최솟값)", "0", "0"]], widths=["22%", "25%", "27%", "26%"], hl=(7, 8), tight=True)}
<div class="grid g2" style="margin-top:10px;font-size:16px;line-height:1.4">
 <div class="card" style="padding:8px 14px"><b style="color:#1c5cab">임계값</b>: 0.05~0.60을 0.025 간격으로 훑어 <b>이웃 임계값까지 평균한 MAE가 가장 낮은 지점</b>을 골랐다(13쪽).</div>
 <div class="card o" style="padding:8px 14px"><b style="color:#c4491c">하한 4(문제 2)</b>: 학습 정답이 모두 4명 이상이라는 가정. 학습 3장이 개선됐고(0.93 → 0.73), 평가에 4명 미만 사진이 있으면 손해.</div>
</div>""", "문제마다 달라진 것은 관점 구성, 임계값, 면적비, 하한 네 가지뿐이고 나머지 규칙은 같다.")

# 11 성능지표 ① 정의
slide("성능지표 ①", "무엇으로, 어떻게 재는가", f"""
<div class="grid g2" style="gap:18px;align-items:start">
<div class="card"><h3>대회 점수</h3>
 <div style="font-size:24px;background:#f6f5f1;border-radius:6px;padding:8px 14px;margin:2px 0 8px">점수 = 100 / (1 + MAE)</div>
 <ul><li>MAE = 이미지별 |예측 수 − 정답 수|의 평균.</li>
 <li>문제 4는 <b>사람 MAE와 차량 MAE의 평균</b>을 쓴다.</li>
 <li>MAE 0이면 100점, 1이면 50점, 2.7이면 약 27점.</li></ul></div>
<div class="card"><h3>MAE ↔ 점수 감각</h3>{tbl(["MAE", "점수"], [["0.33", "75.2"], ["0.50", "66.7"], ["0.73", "57.8"], ["1.00", "50.0"], ["2.73", "26.8"]], nums=(0, 1), widths=["50%", "50%"], tight=True)}</div>
</div>
<div class="grid g3" style="margin-top:10px;gap:12px;font-size:16px;line-height:1.4">
 <div class="card" style="padding:8px 14px"><b style="color:#1c5cab">완전 일치율</b><br>수량을 <b>정확히</b> 맞힌 이미지 비율. 문제 4는 사람·차량 둘 다 맞힌 비율도 본다.</div>
 <div class="card" style="padding:8px 14px"><b style="color:#1c5cab">LOO (한 장씩 빼기)</b><br>29장으로 임계값을 고르고 1장을 평가. 같은 30장으로 임계값을 골라 생긴 <b>과대평가</b> 점검.</div>
 <div class="card o" style="padding:8px 14px"><b style="color:#c4491c">편향 / 표준오차</b><br>편향 = 평균(예측 − 정답), 음수면 <b>덜 센다</b>. MAE 표준오차 약 ±0.12(문제 4) ~ ±0.2~0.4(문제 2).</div>
</div>
<div class="small" style="margin-top:6px;font-size:14px">기준선(템플릿) = <code>imgsz 416, conf 0.25</code> 한 번 추론. 평가 정답은 공개되지 않아 모든 수치는 <b>공개 학습 30장 기준 추정</b>이다.</div>
""", "30장에서는 이미지 1장 = MAE 0.033이라, 0.05 안팎의 차이는 우연일 수 있다.")

# 12 성능지표 ② 결과
slide("성능지표 ②", "결과: 템플릿 대비 MAE는 절반 이하, 점수는 +19~31", f"""
<div class="grid g2" style="gap:16px;margin-bottom:10px">
 <div class="card" style="padding:10px 14px"><div style="font-weight:700">MAE (낮을수록 좋음)</div>{gbars(["문제 2 사람", "문제 4 사람", "문제 4 차량"], [[2.733, 0.933, 0.800], [0.733, 0.333, 0.433]], w=528, h=205, ymax=3.2, fmt="{:.2f}", names=("템플릿(416, conf 0.25)", "최종 메인"))}</div>
 <div class="card" style="padding:10px 14px"><div style="font-weight:700">환산 점수 (높을수록 좋음)</div>{gbars(["문제 2", "문제 4"], [[26.8, 53.6], [57.7, 72.3]], w=528, h=205, ymax=100, fmt="{:.1f}", names=("템플릿", "최종 메인"))}</div>
</div>
{tbl(["", "MAE", "LOO MAE", "완전 일치", "편향", "점수 환산"], [
 ["문제 2 사람", "0.733", "0.733", "0.533", "−0.533", "57.7"],
 ["문제 4 사람", "0.333", "0.333", "0.700", "−0.067", "-"],
 ["문제 4 차량", "0.433", "0.467", "0.600", "−0.300", "-"],
 ["문제 4 평균", "0.383", "0.400", "둘 다 맞힘 0.367 (11/30)", "-", "72.3 (LOO 기준 71.4)"]], nums=(1, 2, 4), hl=(3,), widths=["16%", "11%", "13%", "29%", "11%", "20%"], tight=True)}
""", "두 문제 모두 음수 편향(덜 센다)이 남아 있고, 차량과 문제 2의 편향이 특히 크다.")

# 13 임계값 곡선
slide("성능지표 ③", "임계값을 바꾸면 MAE가 어떻게 변하나 (U자 곡선)", f"""
<div class="grid g21" style="gap:18px;align-items:start">
<div class="card">{lines(["0.075","0.1","0.125","0.15","0.175","0.2","0.225","0.25","0.275","0.3","0.325","0.35"],
 [[1.33,1.07,0.90,0.73,0.83,0.93,1.13,1.20,1.27,1.40,1.53,1.60],[0.60,0.47,0.40,0.33,0.37,0.43,0.47,0.60,0.63,0.83,0.87,0.90],[0.80,0.63,0.53,0.47,0.43,0.47,0.57,0.67,0.77,0.80,0.80,0.77]],
 ["문제 2 사람","문제 4 사람","문제 4 차량"],[BLUE,ORANGE,AQUA],[3,3,4],w=720,h=330,ymax=1.8)}<div class="tiny">가로: 군집 점수 임계값, 세로: 학습 30장 MAE. 원으로 표시한 곳이 최종 선택(0.15, 0.15, 0.175).</div></div>
<div><ul style="font-size:18px;line-height:1.5">
 <li><b>임계값이 낮으면</b> 흐린 박스(오탐·부분 박스)까지 세어 <b>더 센다</b>.</li>
 <li><b>임계값이 높으면</b> 작고 가려진 사람을 버려 <b>덜 센다</b>.</li>
 <li>세 곡선 모두 0.15~0.175 근처에서 바닥이라, 두 문제가 같은 문턱을 쓴다.</li>
 <li>바닥이 뾰족하지 않다: 문제 4 사람은 0.125~0.2에서 0.33~0.43. 한 점이 아니라 <b>이웃 평균(고원)</b>으로 고른 이유.</li>
 <li>문제 2는 바닥이 더 뾰족해(0.73 ↔ 0.83~0.90) 임계값에 더 민감하다.</li></ul></div></div>
""", "임계값은 '덜 세기'와 '더 세기'의 균형점이고, 세 곡선의 바닥이 같은 위치(0.15~0.175)에 모인다.")

# 14 단계별 향상
slide("성능지표 ④", "단계별로 무엇이 MAE를 줄였나", f"""
<div class="grid g3" style="gap:12px">
<div class="card"><h3>문제 2 · 사람</h3>{bars(["템플릿", "단일 해상도|(1280)", "다중 관점|융합", "예측|하한 4"], [2.73, 1.20, 0.97, 0.73], w=334, h=225, ymax=3.2, fmt="{:.2f}", hl=3)}</div>
<div class="card"><h3>문제 4 · 사람</h3>{bars(["템플릿", "단일 해상도|(최선)", "다중 관점|+ 반전"], [0.93, 0.60, 0.33], w=334, h=225, ymax=1.2, fmt="{:.2f}", hl=2)}</div>
<div class="card"><h3>문제 4 · 차량</h3>{bars(["템플릿", "단일 해상도|(최선)", "다중 관점|융합", "면적비|규칙"], [0.80, 0.53, 0.50, 0.43], w=334, h=225, ymax=1.2, fmt="{:.2f}", hl=3)}</div>
</div>
<ul style="margin-top:10px;font-size:18px">
 <li><b>가장 큰 개선은 단일 관점 → 다중 관점 융합</b>이다. 문제 4 사람은 0.60 → 0.33(−45%).</li>
 <li>단일 해상도는 클수록 좋았다: 문제 2는 640 / 960 / 1280에서 1.60 / 1.40 / 1.20, 문제 4 사람 0.60~0.67, 차량 0.53~0.63.</li>
 <li>예측 하한 4는 학습 3장에서 나온 개선(MAE 0.93 → 0.73)이라 가장 가정 의존적이다.</li></ul>
<div class="tiny">템플릿은 CPU 기준(문제 2 2.73, 문제 4 0.93/0.80). 단일 해상도 값은 GPU 실험(문제 4 템플릿은 GPU에서 1.00/0.70). 주황 막대가 최종.</div>
""", "개선의 대부분은 모델을 바꾼 것이 아니라 '여러 번 보고 합치는 방식'에서 나왔다.")

# 15 오류 분석
slide("오류 분석", "어디서 틀리나: 대부분 '덜 센다'", f"""
<div class="grid g2" style="gap:16px">
<div class="card"><div style="font-weight:700;margin-bottom:4px">오답 이미지 수 (정답과 개수가 다른 이미지)</div>{hstack([("문제 2 사람", 11, 3), ("문제 4 사람", 5, 4), ("문제 4 차량", 10, 2)], w=560, row_h=40)}<div class="tiny">문제 2는 GPU 캐시 기준(MAE 0.77, 과소 합 20명·과다 합 3명). CPU 최종(0.733)과 합이 1명 다르다.</div></div>
<div class="card"><div style="font-weight:700;margin-bottom:4px">문제 2 · 정답 수량별 MAE</div>{bars(["4명","5~6명","7~9명","10명+"], [0.14,0.73,1.50,0.83], w=470, h=170, ymax=2.0, fmt="{:.2f}")}<div class="tiny">정답이 7~9명인 이미지가 가장 크게 틀린다. 센 박스의 평균 확신도와 |오차|의 순위 상관 −0.50.</div></div>
</div>
<div class="grid g3" style="margin-top:10px;gap:12px;font-size:14.5px;line-height:1.35">
 <div class="card" style="padding:8px 14px"><b style="color:#1c5cab;font-size:16px">덜 센다(과소)</b><ul style="margin:2px 0;padding-left:18px"><li>수평선·원경의 아주 작은 사람</li><li>군중 속에서 가려지거나 잘린 사람</li><li>유리창 너머·차 안·야간·가장자리의 차</li></ul></div>
 <div class="card o" style="padding:8px 14px"><b style="color:#c4491c;font-size:16px">더 센다(과다)</b><ul style="margin:2px 0;padding-left:18px"><li>마네킹·인형 얼굴, 벽화·사진 속 인물</li><li>길 끝의 작은 차, 군중의 다리</li><li>큰 차량의 부분 박스(면적비로 해결)</li></ul></div>
 <div class="card a" style="padding:8px 14px"><b style="color:#0d7a54;font-size:16px">정답이 0일 때</b><ul style="margin:2px 0;padding-left:18px"><li>사람: 0명 10장 중 오탐 1장</li><li>차량: 0대 8장 중 오탐 0장</li><li>임계값 0.10이면 오탐이 늘고 MAE도 악화</li></ul></div>
</div>""", "오류는 모델이 아예 못 보는 작고 가려진 객체에서 나와 후처리로는 더 줄이기 어렵다.")

# 16 직접 박스 + 미세조정
slide("추가 실험", "박스를 직접 그려 미세조정해 봤더니 오히려 나빠졌다 (문제 4)", f"""
<div class="grid" style="grid-template-columns:290px 290px 1fr;gap:14px;align-items:start">
 <div><img class="fig" src="assets/p4_pseudo_0010.jpg"><div class="tiny">모델 예측 박스(의사 라벨). 수량 일치 11/30 · 이미지 COCO 2017 val</div></div>
 <div><img class="fig" src="assets/p4_hand_0016.jpg"><div class="tiny">직접 보고 고친 박스. 수량 일치 29/30</div></div>
 <div><div style="font-weight:700;margin-bottom:4px">yolov8n 미세조정 (5겹 교차 검증, 학습에 안 쓴 이미지로 평가)</div>{tbl(["설정", "사람", "차량", "평균"], [
  ["<b>사전학습 그대로(메인)</b>", "0.333", "0.433", "0.383"], ["고친 라벨·10층 고정·20에폭", "0.367", "0.533", "0.450"], ["고친 라벨·10층 고정·30에폭", "0.367", "0.567", "0.467"], ["고친 라벨·10층 고정·40에폭", "0.467", "0.567", "0.517"], ["고친 라벨·전체 층·20에폭", "0.500", "0.533", "0.517"], ["의사 라벨·전체 층·20에폭", "0.500", "0.600", "0.550"]], nums=(1, 2, 3), hl=(0,), widths=["52%", "16%", "16%", "16%"])}</div>
</div>
<ul style="margin-top:10px;font-size:17px">
 <li><b>직접 박스</b>: 수량이 안 맞는 19장을 번호·눈금 그림으로 보고 고쳤다(변경 32건 중 확실 13, 추정 19). 박스 128개 중 모델 그대로 107, 후보 채택 7, 직접 그림 14. 사람이 마우스로 그린 것보다 경계가 덜 정확하다.</li>
 <li><b>결과</b>: 모두 사전학습 그대로보다 나쁘고, 에폭을 늘릴수록 나빠졌다. 문제 2도 같은 경향(검토한 박스로 미세조정 1.03~1.17 vs 사전학습 0.77). 24장으로는 사전학습이 이미 아는 것을 개선하기보다 흐트러뜨린다.</li>
 </ul>
""", "30장짜리 박스 라벨로는 사전학습 가중치를 이길 수 없어, 모델은 고정하고 후처리만 최적화했다.")

# 17 채택하지 않은 것
slide("실험 기록", "시도했지만 채택하지 않은 것", f"""
{tbl(["시도", "결과 (MAE)", "판정과 이유"], [
 ["보정 회귀 (임계값별 개수 → 수량)", "문제 2 1.23~2.07 · 문제 4 0.33~0.77 / 0.50~1.03", "기각. 30장으로 회귀하면 과적합"],
 ["관점 추가 (1600, 3×3 타일, 반전)", "문제 2 LOO 0.93~1.17 · 문제 4 0.33 / 0.57", "차이 없음 → 단순한 쪽 유지"],
 ["임계값 아래 후보 확대 재검출", "문제 2 최저 1.20 · 문제 4 차량 0.40(3장 개선)", "기각/미채택. 확대해도 확신이 안 오르거나 중복"],
 ["차량 점수를 클래스 확률 합으로", "문제 4 차량 0.500 (메인 0.433)", "기각. 놓친 차는 확률이 갈린 게 아니라 점수 자체가 낮다"],
 ["해상도·반전·점수 방식 1,035개 전수 탐색", "최저 사람 0.267 / 차량 0.367 (메인 0.333 / 0.433)", "선택 노이즈. 1위와 메인의 차이는 2장"],
 ["밀도 적응 임계값, 점수 합 계수", "사람 0.367·0.333, 차량 0.433·0.433 (LOO는 더 나쁨)", "기각"],
 ["평균 확신도 낮으면 +1 (문제 2)", "학습 MAE 0.73 → 0.60", "미채택. 평가 쪽 확신도가 더 낮아 13장에 +1이 붙을 위험"],
 ["크기별 임계값 2개", "문제 2 0.83(LOO 1.03) · 문제 4 차량 0.40~0.43(LOO 0.47~0.57)", "기각. LOO 악화"],
 ["장면별 규칙 선택 (두 문제 규칙 전환)", "문제 2 0.77~1.17 · 문제 4 0.40~0.60", "개선 없음. 문제별로 따로 두는 것이 최선"]], widths=["31%", "36%", "33%"])}
""", "대부분의 '더 복잡한 방법'은 한 장씩 빼는 검증에서 나빠졌고, 단순한 융합 규칙이 가장 안정적이었다.")

# 18 한계 + 제출
slide("한계와 제출", "주의할 점과 최종 제출", f"""
<div class="grid g2" style="gap:16px">
<div class="card o"><h3>한계 · 위험</h3><ul style="font-size:16px;line-height:1.42">
 <li><b>검증이 30장뿐</b>: 임계값과 규칙을 같은 30장으로 골랐다(LOO로 점검했지만 낙관적). 이미지 1장 = MAE 0.033.</li>
 <li><b>문제 2 하한 4</b>는 학습 정답이 4~13명이라는 가정. 평가에 4명 미만 이미지가 있으면 손해이고, 개선은 학습 3장에서 나왔다.</li>
 <li><b>분포 점검</b>(문제 2 평균): 학습 정답 6.83 / 학습 예측 6.30 / 평가 예측 7.40 → 평가가 더 어렵거나 사람이 많은 것으로 보이나 확인 불가.</li>
 <li>문제 4 차량은 학습에서 평균 0.30대 덜 센다(예측 1.67 / 정답 1.97).</li>
 <li><b>서버 호환 미확인</b>: 서버에는 <code>ultralytics</code>가 없고(torch 1.10), 2코어 실행 시간은 측정하지 않았다.</li></ul></div>
<div class="card"><h3>최종 제출</h3>{tbl(["", "문제 2", "문제 4"], [
 ["컬럼", "<code>filename,person_count</code>", "<code>filename,person_count,<br>vehicle_count</code>"],
 ["행 수", "30 (ID 순서 일치)", "30 (ID 순서 일치)"],
 ["예측 합(평균)", "사람 222 (7.40)", "사람 67 (2.23)<br>차량 50 (1.67)"],
 ["점수 환산", "57.7", "72.3 (LOO 71.4)"],
 ["메인 버전", "v3 예측 하한 4", "v3 차량 면적비 0.1"]], widths=["24%", "34%", "42%"], tight=True)}
 <div class="small" style="margin-top:8px">노트북 <code>work.ipynb</code>를 처음부터 실행(Restart & Run All)해 출력까지 저장, 오류 없음. CPU 약 33~41초(개발 PC).</div></div>
</div>
""", "사전학습 YOLOv8n을 '여러 관점으로 보고 합치는' 규칙만으로 템플릿 대비 점수를 +31(문제 2)·+19(문제 4) 올렸다.")

# ---------------- HTML 조립 ----------------
def render():
    n = len(S); out = []
    for i, s in enumerate(S, 1):
        if s["cover"]: out.append(f'<section class="slide">{s["body"]}</section>'); continue
        out.append(f'<section class="slide"><div class="kick">{esc(s["kick"])}</div><h1>{s["title"]}</h1><div class="body">{s["body"]}</div><div class="take"><b>한 줄 결론</b>{s["take"]}</div><div class="foot"><span>{DATE}</span><span>{TITLE_FOOT}</span><span>{i} / {n}</span></div></section>')
    js = "<script>const q=new URLSearchParams(location.search).get('s');if(q){document.querySelectorAll('.slide').forEach((e,i)=>{if(i+1!=q)e.style.display='none'})}</script>"
    return f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>문제 2·4 사진 속 인원·차량 집계 발표자료</title><style>{CSS}</style></head><body>{"".join(out)}{js}</body></html>'

if __name__ == "__main__":
    p = os.path.join(HERE, "deck.html"); open(p, "w", encoding="utf-8").write(render()); print("slides", len(S), "->", p)
