cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
while wmic process where "name='python.exe'" get CommandLine 2>/dev/null | grep -v wmic | grep -q "_run_cv.py"; do sleep 4; done
python _run_cv.py s 20 dihedral+synth 1 0.0002 > _cv2_dihedral+synth_f1.log 2>&1
for cfg in base dihedral synth dihedral+synth; do
  python _run_cv.py s 20 $cfg 2 0.0002 > "_cv2_${cfg}_f2.log" 2>&1
done
echo ALLDONE > _cv_done3.flag
