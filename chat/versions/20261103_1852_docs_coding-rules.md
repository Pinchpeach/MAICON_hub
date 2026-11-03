# 2026-11-03 18:52 KST · docs · 코딩 규칙 추가 (rules/Coding rules.md)

- 작성: claude2 · 관련 chat: 003 · 커밋: 011e6c7
- 배경: 사용자 새 규칙(11.03 18:51) — 규칙을 md 파일로 나누어 설정, 'Coding rules.md'에 코딩 행동 규칙.

## 바뀐 파일

| 파일 | 내용 |
|---|---|
| `rules/Coding rules.md` (새 파일) | 사용자가 준 원문 그대로(영문, 4개 장: Think Before Coding / Simplicity First / Surgical Changes / Goal-Driven Execution) |
| `CLAUDE.md` (새 파일, 루트) | 에이전트가 시작할 때 읽을 규칙 파일 목록(rules/Coding rules.md, chat/chat_room.md 머리말, chat/versions/README.md). Claude Code는 루트 CLAUDE.md를 자동으로 읽으므로 진입점으로 둠 |
| `chat/versions/README.md` | 목록에 이 기록 추가 |
| `README.md` | '큰 변경' 표에 한 줄, '폴더' 표에 rules/·CLAUDE.md 추가 |
| `chat/chat_room.md` | 003번 글 |

## 영향

- 코드 변경 없음. 이후 모든 구현은 Coding rules를 따름: 가정 명시·모호하면 질문, 요청 밖 기능·추상화 금지, 요청과 관계없는 코드 수정 금지, 작업마다 검증 기준을 먼저 정하고 확인.

## 검증

- 원문과 같은지 눈으로 대조. 파일 생성·diff 확인.

## 되돌리기

- 이 커밋을 `git revert`
