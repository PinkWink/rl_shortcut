#!/usr/bin/env bash
# 강사에게 받은 학습된 모델을 다운로드 폴더에서 작업 폴더의 models/ 로 가져온다 (학습을 끝내지 못했을 때).
#   cd ~/rl_shortcut
#   ./scripts/get_model.sh go2.pt
# 받을 수 있는 이름: pendulum.pt · pinky_road_seg.pt · pinky.pt · go2.pt · g1_walk.pt · g1_bruce_lee.pt
# 가져온 뒤에는 play.py 에 --checkpoint models/<이름> 을 붙여 실행한다 (YOLO 는 --weights models/pinky_road_seg.pt).
set -eo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"
NAME="${1:-}"
if [[ -z "$NAME" ]]; then
  echo "사용법: ./scripts/get_model.sh <모델 파일 이름>     예) ./scripts/get_model.sh go2.pt"
  echo "이름: pendulum.pt · pinky_road_seg.pt · pinky.pt · go2.pt · g1_walk.pt · g1_bruce_lee.pt"
  exit 1
fi
NAME="$(basename "${NAME%.pt}").pt"
STEM="${NAME%.pt}"

# 다운로드 폴더: 한글 Ubuntu 는 ~/다운로드, 영문은 ~/Downloads
DIRS=()
D="$(xdg-user-dir DOWNLOAD 2>/dev/null || true)"
for d in "$D" "$HOME/Downloads" "$HOME/다운로드"; do
  [[ -n "$d" && "$d" != "$HOME" && -d "$d" ]] && DIRS+=("$d")
done
# 같은 파일을 두 번 받으면 브라우저가 "go2 (1).pt" 처럼 이름을 바꾼다: 그중 가장 최근 것을 고른다.
FOUND=""
for d in "${DIRS[@]}"; do
  for f in "$d/$STEM".pt "$d/$STEM"\ \(*\).pt "$d/$STEM"\(*\).pt; do
    [[ -f "$f" ]] || continue
    [[ -z "$FOUND" || "$f" -nt "$FOUND" ]] && FOUND="$f"
  done
done
if [[ -z "$FOUND" ]]; then
  echo "❌ 다운로드 폴더에 $NAME 이 없습니다. 찾아본 곳: ${DIRS[*]:-(다운로드 폴더 없음)}"
  echo "   강사가 배포한 $NAME 을 먼저 내려받은 뒤 다시 실행하세요."
  exit 1
fi
[[ -s "$FOUND" ]] || { echo "❌ $FOUND 의 크기가 0 입니다. 다시 내려받으세요."; exit 1; }

mkdir -p models
cp -f "$FOUND" "models/$NAME"
echo "가져왔습니다: $FOUND"
echo "        →  $WS/models/$NAME  ($(du -h "models/$NAME" | cut -f1))"
echo
echo "이 모델로 실행하기 (source scripts/env.sh 를 한 터미널에서):"
case "$STEM" in
  pendulum)       echo "  python scripts/play.py --task pendulum --checkpoint models/$NAME" ;;
  pinky)          echo "  python scripts/play.py --task pinky --checkpoint models/$NAME" ;;
  go2)            echo "  python scripts/play.py --task go2 --checkpoint models/$NAME" ;;
  g1_walk)        echo "  python scripts/play.py --task g1 --motion walk --checkpoint models/$NAME" ;;
  g1_bruce_lee)   echo "  python scripts/play.py --task g1 --motion bruce_lee --checkpoint models/$NAME" ;;
  pinky_road_seg) echo "  python scripts/pinky_yolo.py --check --weights models/$NAME"
                  echo "  python scripts/play.py --task pinky --num_envs 1 --perception yolo --enable_cameras --weights models/$NAME" ;;
  *)              echo "  python scripts/play.py --task <로봇> --checkpoint models/$NAME" ;;
esac
