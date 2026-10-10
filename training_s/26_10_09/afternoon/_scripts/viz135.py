import sys, os, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from cv135b import Tobit

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # 검증된 범주 색 1~3번 (train / 강조·fit / test)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": True, "legend.facecolor": SURF, "legend.edgecolor": SURF, "legend.framealpha": 0.92, "legend.fontsize": 8.5})
OUT = "_viz"; os.makedirs(OUT, exist_ok=True)
L = lambda p, f="train": pd.read_csv(f"problem{p}/data/{f}.csv")
dot = dict(s=34, edgecolors=SURF, linewidths=1.2, zorder=3)

def hist_tt(ax, tr, te, name, bins=12):
    """feature 분포: train(막대) vs test(개별 값 눈금). test 는 20행뿐이라 막대 대신 값 위치로 표시"""
    lo, hi = min(tr.min(), te.min()), max(tr.max(), te.max()); b = np.linspace(lo, hi, bins + 1)
    ax.hist(tr, bins=b, color=BLUE, alpha=0.85, rwidth=0.92, label=f"train ({len(tr)}행)", zorder=2)
    ax.plot(te, np.full(len(te), -0.045 * ax.get_ylim()[1]), "|", color=ORANGE, markersize=11, markeredgewidth=1.8, label=f"test ({len(te)}행) 값 위치", clip_on=False, zorder=4)
    ax.set_title(f"{name} 분포"); ax.set_xlabel(f"train 평균 {tr.mean():.2f} · test 평균 {te.mean():.2f}"); ax.set_ylabel("행 수"); ax.grid(axis="x", visible=False)

# =========================== P1 ===========================
t, te = L("01"), L("01", "test"); F = ["avg_sleep_hours", "night_shift_days", "caffeine_intake"]; KO = ["평균 수면 시간", "야간근무 일수", "카페인 섭취량(mg)"]
y = t.fatigue_score.values; cen = y <= 30
fig, ax = plt.subplots(3, 3, figsize=(14, 11.5)); fig.suptitle("문제 1 · 수면 기반 피로도 — 데이터와 feature 상세", x=0.01, ha="left", fontsize=14, fontweight="bold")
for j, (c, k) in enumerate(zip(F, KO)):
    a = ax[0, j]; a.scatter(t[c][~cen], y[~cen], color=BLUE, label="30 초과 (65행)", **dot); a.scatter(t[c][cen], y[cen], color=ORANGE, marker="s", label="하한 30에 걸린 행 (15행)", **dot)
    xs = np.linspace(t[c].min(), t[c].max(), 50); m = LinearRegression().fit(t[[c]][~cen], y[~cen]); a.plot(xs, m.predict(xs.reshape(-1, 1)), color=INK2, lw=1.6, zorder=2)
    a.axhline(30, color=INK2, lw=0.8); a.set_title(f"{k} vs 피로도  (상관 {np.corrcoef(t[c], y)[0, 1]:+.2f})"); a.set_xlabel(k); a.set_ylabel("fatigue_score")
    if j == 0: a.legend(loc="upper right")
    hist_tt(ax[1, j], t[c], te[c], k, bins=6 if c == "night_shift_days" else 12)
    if j == 0: ax[1, j].legend(loc="upper right")
a = ax[2, 0]; a.hist(y, bins=np.arange(30, 72, 2.5), color=BLUE, rwidth=0.92, zorder=2); a.set_title("타깃 fatigue_score 분포"); a.set_xlabel("30.0 에 15행(19%)이 몰려 있음 → max(30, 선형+잡음) 구조"); a.set_ylabel("행 수"); a.grid(axis="x", visible=False)
a.annotate("하한 30.0\n15행", xy=(31.2, 15), xytext=(40, 14), fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
oof_l, oof_t = np.zeros(80), np.zeros(80)
for tr_i, va_i in KFold(5, shuffle=True, random_state=0).split(t):
    oof_l[va_i] = LinearRegression().fit(t[F].iloc[tr_i], y[tr_i]).predict(t[F].iloc[va_i]); oof_t[va_i] = Tobit().fit(t[F].iloc[tr_i], y[tr_i]).predict(t[F].iloc[va_i])
for a, o, nm in ((ax[2, 1], oof_l, "단순 선형 회귀"), (ax[2, 2], oof_t, "Tobit (메인)")):
    a.scatter(o[~cen], y[~cen], color=BLUE, label="30 초과", **dot); a.scatter(o[cen], y[cen], color=ORANGE, marker="s", label="하한 30", **dot); a.plot([14, 70], [14, 70], color=INK2, lw=1)
    a.set_title(f"{nm}: 교차검증 예측 vs 정답  (RMSE {np.sqrt(np.mean((o - y) ** 2)):.2f})"); a.set_xlabel("예측값 (5-Fold OOF, 분할 1회)"); a.set_ylabel("정답"); a.set_xlim(14, 70); a.set_ylim(27, 72)
ax[2, 1].legend(loc="upper left"); ax[2, 1].annotate("30 미만으로 예측 →\n하한 행에서 큰 오차", xy=(22, 30), xytext=(16, 45), fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(f"{OUT}/p1_fatigue.png", dpi=115); plt.close(fig)

# =========================== P3 ===========================
t, te = L("03"), L("03", "test"); F = ["leadership_score", "facility_score", "meal_score", "rest_time_score"]; KO = ["리더십 점수", "시설 점수", "급식 점수", "휴식시간 점수"]; y = t.overall_satisfaction.values
fig, ax = plt.subplots(3, 4, figsize=(16, 11.5)); fig.suptitle("문제 3 · 복무 만족도 — 데이터와 feature 상세", x=0.01, ha="left", fontsize=14, fontweight="bold")
lm = LinearRegression().fit(t[F], y)
for j, (c, k) in enumerate(zip(F, KO)):
    a = ax[0, j]; a.scatter(t[c], y, color=BLUE, **dot); xs = np.linspace(t[c].min(), t[c].max(), 50); m = LinearRegression().fit(t[[c]], y); a.plot(xs, m.predict(xs.reshape(-1, 1)), color=INK2, lw=1.6, zorder=2)
    a.set_title(f"{k} vs 종합 만족도  (상관 {np.corrcoef(t[c], y)[0, 1]:+.2f})"); a.set_xlabel(f"{k}  · 다중회귀 계수 {lm.coef_[j]:.3f}"); a.set_ylabel("overall_satisfaction")
    hist_tt(ax[1, j], t[c], te[c], k)
    if j == 0: ax[1, j].legend(loc="upper left")
a = ax[2, 0]; a.hist(y, bins=np.arange(4.0, 8.8, 0.4), color=BLUE, rwidth=0.92, zorder=2); a.set_title("타깃 overall_satisfaction 분포"); a.set_xlabel("범위 4.1 ~ 8.4, 0.1 단위"); a.set_ylabel("행 수"); a.grid(axis="x", visible=False)
a = ax[2, 1]; mean4 = t[F].mean(axis=1); a.scatter(mean4, y, color=BLUE, **dot); xs = np.linspace(mean4.min(), mean4.max(), 50); a.plot(xs, LinearRegression().fit(mean4.values.reshape(-1, 1), y).predict(xs.reshape(-1, 1)), color=INK2, lw=1.6)
a.set_title(f"4항목 평균 vs 종합 만족도  (상관 {np.corrcoef(mean4, y)[0, 1]:+.2f})"); a.set_xlabel("네 점수의 단순 평균"); a.set_ylabel("overall_satisfaction")
oof = np.zeros(80)
for tr_i, va_i in KFold(5, shuffle=True, random_state=0).split(t): oof[va_i] = LinearRegression().fit(t[F].iloc[tr_i], y[tr_i]).predict(t[F].iloc[va_i])
a = ax[2, 2]; a.scatter(oof, y, color=BLUE, **dot); a.plot([4, 8.6], [4, 8.6], color=INK2, lw=1); a.set_title(f"선형 회귀: 교차검증 예측 vs 정답  (RMSE {np.sqrt(np.mean((oof - y) ** 2)):.3f})"); a.set_xlabel("예측값 (5-Fold OOF, 분할 1회)"); a.set_ylabel("정답")
a = ax[2, 3]; r = y - oof; a.hist(r, bins=14, color=BLUE, rwidth=0.92, zorder=2); a.set_title(f"잔차 분포  (표준편차 {r.std():.2f})"); a.set_xlabel("정답 - 예측. 다른 변수로 설명되는 구조를 찾지 못함"); a.set_ylabel("행 수"); a.grid(axis="x", visible=False)
fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(f"{OUT}/p3_satisfaction.png", dpi=115); plt.close(fig)

# =========================== P5 ===========================
t, te = L("05"), L("05", "test"); F = ["pushup", "run_time", "sleep_quality", "stress_level"]; KO = ["팔굽혀펴기(회)", "달리기 시간(분)", "수면의 질", "스트레스 수준"]; fit = (t.status == "fit").values
HI = {"pushup": 38, "run_time": 9.6, "sleep_quality": 5.2, "stress_level": 5.8}; LO = {"pushup": 35, "run_time": 11.0, "sleep_quality": 4.0, "stress_level": 6.0}; DIR = {"pushup": "≥", "run_time": "≤", "sleep_quality": "≥", "stress_level": "≤"}
fig, ax = plt.subplots(3, 4, figsize=(16, 12)); fig.suptitle("문제 5 · 전투 적합 여부·회복일수 — 데이터와 feature 상세", x=0.01, ha="left", fontsize=14, fontweight="bold")
rng = np.random.default_rng(0)
for j, (c, k) in enumerate(zip(F, KO)):
    a = ax[0, j]; a.axvspan(min(HI[c], LO[c]), max(HI[c], LO[c]), color=GRID, zorder=1, label="임계값이 있을 수 있는 구간")
    a.scatter(t[c][~fit], rng.uniform(-0.28, 0.28, (~fit).sum()), color=BLUE, label="unfit (70명)", **dot); a.scatter(t[c][fit], 1 + rng.uniform(-0.28, 0.28, fit.sum()), color=ORANGE, marker="s", label="fit (10명)", **dot)
    a.axvline(HI[c], color=INK, lw=1.2, zorder=2); a.set_yticks([0, 1]); a.set_yticklabels(["unfit", "fit"]); a.set_ylim(-0.6, 1.75); a.grid(axis="y", visible=False)
    a.set_title(f"{k}: status 별 값"); a.set_xlabel(f"사용 임계값 {DIR[c]} {HI[c]}  (실제는 {min(HI[c], LO[c])}~{max(HI[c], LO[c])} 사이)")
    if j == 0: a.legend(loc="upper left", ncol=1, fontsize=8)
    hist_tt(ax[1, j], t[c], te[c], k)
    if j == 0: ax[1, j].legend(loc="upper right")
def box_panel(a, cx, cy, kx, ky):
    a.scatter(t[cx][~fit], t[cy][~fit], color=BLUE, label="train unfit", **dot); a.scatter(t[cx][fit], t[cy][fit], color=ORANGE, marker="s", label="train fit", **dot)
    a.scatter(te[cx], te[cy], color=AQUA, marker="^", label="test (정답 없음)", s=62, edgecolors=SURF, linewidths=1.2, zorder=4)
    a.axvline(HI[cx], color=INK, lw=1.1); a.axhline(HI[cy], color=INK, lw=1.1); a.set_xlabel(f"{kx}  (기준 {DIR[cx]} {HI[cx]})"); a.set_ylabel(f"{ky}  (기준 {DIR[cy]} {HI[cy]})")
    for sid in ("S035", "S066"):
        r = te[te.soldier_id == sid].iloc[0]; a.annotate(sid, (r[cx], r[cy]), textcoords="offset points", xytext=(7, 6), fontsize=9, fontweight="bold")
box_panel(ax[2, 0], "run_time", "pushup", "달리기 시간", "팔굽혀펴기"); ax[2, 0].set_title("달리기 × 팔굽혀펴기 (왼쪽 위가 통과 영역)"); ax[2, 0].legend(loc="lower left", fontsize=8)
box_panel(ax[2, 1], "stress_level", "sleep_quality", "스트레스", "수면의 질"); ax[2, 1].set_title("스트레스 × 수면의 질 (왼쪽 위가 통과 영역)")
a = ax[2, 2]; b = np.arange(4.5, 17.5, 1); a.hist([t.recovery_days[~fit], t.recovery_days[fit]], bins=b, color=[BLUE, ORANGE], label=["unfit (평균 10.07일)", "fit (평균 5.50일)"], rwidth=0.9, zorder=2)
a.set_title("회복일수 분포: status 가 거의 결정"); a.set_xlabel("recovery_days"); a.set_ylabel("행 수"); a.legend(loc="upper right"); a.grid(axis="x", visible=False)
a = ax[2, 3]; u = t[~fit]; a.scatter(u.stress_level, u.recovery_days + rng.uniform(-0.15, 0.15, len(u)), color=BLUE, **dot); xs = np.linspace(u.stress_level.min(), u.stress_level.max(), 20)
a.plot(xs, LinearRegression().fit(u[["stress_level"]], u.recovery_days).predict(xs.reshape(-1, 1)), color=INK2, lw=1.6); a.set_title(f"unfit 안에서 회복일수 vs 스트레스  (상관 {u.stress_level.corr(u.recovery_days):+.2f})"); a.set_xlabel("가장 상관이 큰 feature 도 약함 → 그룹 평균 사용"); a.set_ylabel("recovery_days")
fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(f"{OUT}/p5_fit_recovery.png", dpi=115); plt.close(fig)
print("saved", os.listdir(OUT))
