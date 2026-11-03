# 2026-11-03 18:26 KST · docs · chat room 이전 + 2026 본선 형식

- 작성: claude2 · 관련 chat: 001 · 커밋: 3c2def7
- 배경: 사용자 결정(11.02 20:28) 리셋 — 이전 작업(Drive chat 1~414, 코드 v0.1.0~v0.5.67)은 실험 결과로 보관. 사용자 결정(11.03 18:24) 작업 레포는 Pinchpeach/MAICON_hub, `chat/` 폴더에서 기존 채팅 방식 유지.

## 바뀐 파일

| 파일 | 내용 |
|---|---|
| `chat/chat_room.md` (새 파일) | 쓰는 법(머리줄 형식 `NNN \| 시각 [제목] - 서명`, 댓글 형식, 번호 겹침 처리, 쓴 뒤 commit+push), 역할(Claude 구현 / claude2 검증 / 사용자 큰 결정), 001번 글(리셋·레포 이전·2026 본선 형식 요약) |

## 영향

- 코드 변경 없음. 공식 chat room이 Drive `MAICON_대회용/chat room.md`(글 416까지, 보관)에서 이 파일로 바뀜.
- 작업용 브랜치 `chat-room`을 만들었다가 main에 fast-forward로 합침. 원격 `chat-room` 브랜치는 이 환경에서 삭제가 막혀 남아 있음(main과 같은 커밋).

## 검증

- main HEAD 3c2def7에 `chat/chat_room.md` 존재 확인(git ls-remote).

## 되돌리기

- `git revert 3c2def7`
