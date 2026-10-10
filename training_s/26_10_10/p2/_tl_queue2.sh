cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
for cfg in "n n_e15_640 15 640 8" "s s_e15_640 15 640 8" "s s_e30_640 30 640 8"; do
  set -- $cfg; python _train_tl.py $1 $2 $3 $4 $5 > "_tl_$2.log" 2>&1
done
echo ALLDONE > _tl_done2.flag
