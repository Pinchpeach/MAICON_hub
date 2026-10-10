cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
for k in 0 1 2 3 4; do
  for cfg in base dihedral synth dihedral+synth; do
    python _run_cv.py s 20 $cfg $k 0.0002 > "_cv2_${cfg}_f$k.log" 2>&1
  done
done
echo ALLDONE > _cv_done2.flag
