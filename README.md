# MAICON_hub
Learning &amp; Training for MAICON and Project in 5corps

2026 국방 AI 경진대회(MAICON) 본선 준비 레포. 본선 과제: AI 기반 UGV 도심 전술 작전, 3:3 대결 (2026.11.18~11.20).

## 현재 코드 상태 (2026-11-03 18:30 KST)

- **2026 본선용 로봇 코드: 아직 없음.** 2차 멘토링(11.04) 세부 규정을 받은 뒤 새 구조 설계부터 시작한다.
- 리셋 전 코드(2025 형식 기준 v0.5.67 — 차선 주행·포트홀 회피·정찰 집계, 시뮬 평가 544 자연 지도 중 532 clean)는 이 레포에 없고 실험 결과로만 보관(Google Drive `MAICON_대회용`, Claude 프로젝트 `claude/Experiment_Results_v0.5.67.md`).
- 이 레포에 있는 것: 예선·연습용 학습 자료(`MAICON_pre/`, `training_s/`, `Kaggle/`), 개발 컨테이너(`docker11/`), 작업 기록(`chat/`).

## 큰 변경

| 시각(KST) | 변경 | 기록 |
|---|---|---|
| 2026-11-02 20:28 | 프로젝트 리셋 — 이전 작업(v0.1.0~v0.5.67)은 실험 결과로 보관 | (리셋 전 Drive chat 415) |
| 2026-11-03 18:26 | 작업 레포를 MAICON_hub로, 소통은 `chat/chat_room.md`(main에 직접 커밋) | `chat/versions/20261103_1826_docs_chat-room-init.md` |
| 2026-11-03 18:30 | 변경 기록 규칙: `chat/versions/`에 시간별 상세 기록, 이 README에 큰 변경·현재 코드 상태 | `chat/versions/20261103_1830_docs_versions-readme-rules.md` |
| 2026-11-03 18:52 | 규칙을 주제별 md로 분리: `rules/Coding rules.md`(코딩 행동 규칙), 진입점 `CLAUDE.md` | `chat/versions/20261103_1852_docs_coding-rules.md` |

## 폴더

| 폴더 | 내용 |
|---|---|
| `CLAUDE.md` | 에이전트 규칙 목록(시작 때 읽을 파일) |
| `rules/` | 주제별 규칙 md (`Coding rules.md`) |
| `chat/chat_room.md` | 에이전트·사용자 작업 대화 (번호 글, 규칙은 파일 머리) |
| `chat/versions/` | 변경 1건마다 상세 기록 (`YYYYMMDD_HHMM_<버전>_<제목>.md`, 목록은 `chat/versions/README.md`) |
| `MAICON_pre/` | 24·25 MAICON 기출 문제와 풀이 |
| `training_s/` | 학습·실험 기록과 스니펫 |
| `Kaggle/` | Kaggle 연습 |
| `docker11/` | 개발 컨테이너(Dockerfile, compose, 에이전트 규칙) |
| `DATA_SOURCES.md` | 용량 문제로 빠진 데이터의 출처·복원 방법 |
