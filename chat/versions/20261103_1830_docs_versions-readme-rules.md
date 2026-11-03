# 2026-11-03 18:30 KST · docs · versions 폴더·루트 README 규칙 추가

- 작성: claude2 · 관련 chat: 002 · 커밋: 34faa57
- 배경: 사용자 추가 규칙(11.03 18:29)
  1. `chat/` 아래 `versions/` 폴더를 만들고 시간별 변경을 상세히 기록
  2. 루트 `README.md`에 큰 변경과 현재 코드 상태를 기록·수정

## 바뀐 파일

| 파일 | 내용 |
|---|---|
| `chat/versions/README.md` (새 파일) | 기록 규칙(파일 이름 `YYYYMMDD_HHMM_<버전>_<제목>.md`, 적을 6가지 항목, 같은 커밋에 넣기, 큰 변경은 루트 README도 수정) + 기록 목록 표 |
| `chat/versions/20261103_1826_docs_chat-room-init.md` (새 파일) | 커밋 3c2def7 소급 기록 |
| `chat/versions/20261103_1827_docs_main-direct.md` (새 파일) | 커밋 53e179f 소급 기록 |
| `chat/versions/20261103_1830_docs_versions-readme-rules.md` (새 파일) | 이 기록 |
| `chat/chat_room.md` | '쓰는 법'에 versions 기록·루트 README 수정 규칙 2줄 추가, 002번 글 |
| `README.md` | 레포 소개 + '현재 코드 상태'(2026 본선용 코드 아직 없음) + '큰 변경' 표 + 폴더 안내로 다시 씀 |

## 영향

- 코드 변경 없음. 이후 모든 변경은 versions 기록이 필수, 큰 변경은 루트 README도 수정.

## 검증

- 파일 생성·diff 확인만.

## 되돌리기

- 이 커밋을 `git revert`
