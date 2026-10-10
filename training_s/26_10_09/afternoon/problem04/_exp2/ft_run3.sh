cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
until [ -f _exp2/ft_done2.flag ]; do sleep 5; done
run() { if [ "$(date +%H%M)" -lt 1623 ]; then python _exp2/ft_cv.py "$@" > "_exp2/ft_$1.log" 2>&1; else echo "SKIP $1 (시간 초과)" >> _exp2/ft_run3.log; fi; }
run hand_frz10_e30_lowlr _exp2/dummy_boxes_hand/labels 30 10 0.0002
run hand_frz20_e30 _exp2/dummy_boxes_hand/labels 30 20
run hand_frz10_e10 _exp2/dummy_boxes_hand/labels 10 10
run hand_frz20_e40_lowlr _exp2/dummy_boxes_hand/labels 40 20 0.0002
run hand_all_e30_lowlr _exp2/dummy_boxes_hand/labels 30 0 0.0001
echo ALLDONE > _exp2/ft_done3.flag
