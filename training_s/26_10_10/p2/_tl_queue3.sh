cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
run() { n=$1; shift; python _train_tl.py s "$n" "$@" > "_tl_$n.log" 2>&1; }
run s_e40_640 40 640 8
run s_e50_640 50 640 8
run s_e60_640 60 640 8
run s_e80_640 80 640 8
run s_e60_640_lr1e3 60 640 8 _tl_data.yaml '{"optimizer":"AdamW","lr0":0.001}'
run s_e60_640_lr5e4 60 640 8 _tl_data.yaml '{"optimizer":"AdamW","lr0":0.0005}'
run s_e60_640_lr2e4 60 640 8 _tl_data.yaml '{"optimizer":"AdamW","lr0":0.0002}'
echo ALLDONE > _tl_done3.flag
