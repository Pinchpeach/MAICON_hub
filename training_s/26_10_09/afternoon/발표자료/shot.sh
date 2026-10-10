#!/bin/bash
# 사용: bash shot.sh 3 5 7  (슬라이드 번호별 PNG 를 _shots/ 에 저장)
cd "$(dirname "$0")"; mkdir -p _shots
EDGE="C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"; H="file:///$(pwd -W | sed 's# #%20#g')/deck.html"
for n in "$@"; do "$EDGE" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=1 --window-size=1280,720 --screenshot="$(pwd -W)/_shots/s$n.png" "$H?s=$n" >/dev/null 2>&1; done
