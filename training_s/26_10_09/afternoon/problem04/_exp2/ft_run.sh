cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
python _exp2/ft_cv.py hand_all_e20 _exp2/dummy_boxes_hand/labels 20 0 > _exp2/ft_hand_all_e20.log 2>&1
python _exp2/ft_cv.py hand_frz10_e20 _exp2/dummy_boxes_hand/labels 20 10 > _exp2/ft_hand_frz10_e20.log 2>&1
python _exp2/ft_cv.py pseudo_all_e20 _exp2/dummy_boxes/labels 20 0 > _exp2/ft_pseudo_all_e20.log 2>&1
echo ALLDONE > _exp2/ft_done.flag
