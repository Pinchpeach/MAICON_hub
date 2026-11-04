# MAICON_hub
Learning &amp; Training for MAICON and Project in 5corps

2026 국방 AI 경진대회(MAICON) 본선 준비 레포.

## 현재 코드 상태 (2026-11-04 20:18 KST)

- **2026 본선용 로봇 코드: 아직 없음.** 사용자 결정(11.04): 구조 설계 초안대로 진행, 로봇은 본선 전에 받을 수 없어 **시뮬(가상 환경)에서 개발**. 첫 목표는 시뮬에서 로봇 장치 제어 확인 → 표적 조준·레이저 발사 한 바퀴.
- 대회 규칙·설계 문서는 대회 자료 배포 금지에 따라 이 공개 레포에 두지 않고 Claude 프로젝트(비공개)에만 둔다. 이 레포에는 코드와 작업 기록만.
- 리셋 전 코드(2025 형식 기준 v0.5.67)는 이 레포에 없고 실험 결과로만 보관(Google Drive `MAICON_대회용`, Claude 프로젝트 `claude/Experiment_Results_v0.5.67.md`).
- 이 레포에 있는 것: 예선·연습용 학습 자료(`MAICON_pre/`, `training_s/`, `Kaggle/`), 개발 컨테이너(`docker11/`), 규칙(`CLAUDE.md`, `rules/`), 작업 기록(`chat/`).

## 큰 변경

| 시각(KST) | 변경 | 기록 |
|---|---|---|
| 2026-11-02 20:28 | 프로젝트 리셋 — 이전 작업(v0.1.0~v0.5.67)은 실험 결과로 보관 | (리셋 전 Drive chat 415) |
| 2026-11-03 18:26 | 작업 레포를 MAICON_hub로, 소통은 `chat/chat_room.md`(main에 직접 커밋) | `chat/versions/20261103_1826_docs_chat-room-init.md` |
| 2026-11-03 18:30 | 변경 기록 규칙: `chat/versions/`에 시간별 상세 기록, 이 README에 큰 변경·현재 코드 상태 | `chat/versions/20261103_1830_docs_versions-readme-rules.md` |
| 2026-11-03 18:52 | 규칙을 주제별 md로 분리: `rules/Coding rules.md`(코딩 행동 규칙) + `rules/realworld examples.md`(실제 코드 예시), 진입점 `CLAUDE.md` | `chat/versions/20261103_1852_docs_coding-rules.md` |
| 2026-11-04 20:18 | 2026 본선 개발 방향 확정: 설계 초안대로, 시뮬에서 개발, 대회 문서는 비공개 프로젝트에만 | `chat/versions/20261104_2018_docs_plan-sim-private.md` |

## 폴더

| 폴더 | 내용 |
|---|---|
| `CLAUDE.md` | 에이전트 규칙 목록(시작 때 읽을 파일) |
| `rules/` | 주제별 규칙 md (`Coding rules.md`, `realworld examples.md`) |
| `chat/chat_room.md` | 에이전트·사용자 작업 대화 (번호 글, 규칙은 파일 머리) |
| `chat/versions/` | 변경 1건마다 상세 기록 (`YYYYMMDD_HHMM_<버전>_<제목>.md`, 목록은 `chat/versions/README.md`) |
| `MAICON_pre/` | 24·25 MAICON 기출 문제와 풀이 |
| `training_s/` | 학습·실험 기록과 스니펫 |
| `Kaggle/` | Kaggle 연습 |
| `docker11/` | 개발 컨테이너(Dockerfile, compose, 에이전트 규칙) |
| `DATA_SOURCES.md` | 용량 문제로 빠진 데이터의 출처·복원 방법 |
