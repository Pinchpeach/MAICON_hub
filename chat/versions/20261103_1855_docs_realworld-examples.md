# 2026-11-03 18:55 KST · docs · 실제 코드 예시 규칙 추가 (rules/realworld examples.md)

- 작성: claude2 · 관련 chat: 003 댓글 · 커밋: 같은 커밋(해시는 chat 003 댓글에 적음)
- 배경: 사용자 새 규칙(11.03 18:51) — 'realworld examples.md', 실제 코드 예시 규칙. Coding rules와 함께 md로 나눠 둠.

## 바뀐 파일

| 파일 | 내용 |
|---|---|
| `rules/realworld examples.md` (새 파일) | 사용자 첨부 원문 그대로(14,695 bytes, cmp로 동일 확인). Coding rules 4개 원칙별 예시: 숨은 가정·여러 해석 / 과한 추상화·추측성 기능 / 김에 리팩터링·스타일 바꾸기 / 막연한 목표 vs 검증 가능한 목표·단계별 검증·시험 먼저. 끝에 안티패턴 요약과 핵심 |
| `CLAUDE.md` | 규칙 목록 표에 한 줄 추가 |
| `README.md` | '큰 변경' 18:52 줄과 '폴더'의 rules/ 설명에 파일 이름 추가 |
| `chat/versions/README.md` | 목록에 이 기록 추가 |
| `chat/chat_room.md` | 003번 글에 댓글 |

## 영향

- 코드 변경 없음. 구현 때 Coding rules를 이 예시 기준으로 해석.

## 검증

- `cmp`로 첨부 원문과 바이트 단위 동일.

## 되돌리기

- 이 커밋을 `git revert`
