cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
for cfg in "n n_e60_640 60 640 8" "n n_e100_640 100 640 8" "s s_e30_640 30 640 8" "s s_e60_640 60 640 8" "s s_e100_640 100 640 8" "s s_e60_960 60 960 8"; do
  set -- $cfg; python _train_tl.py $1 $2 $3 $4 $5 > "_tl_$2.log" 2>&1
done
echo ALLDONE > _tl_done.flag
