cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
python _run_cv.py n 30 base 1,2,3,4 > _cv_base.log 2>&1
python _run_cv.py n 30 dihedral > _cv_dihedral.log 2>&1
python _run_cv.py n 30 synth > _cv_synth.log 2>&1
python _run_cv.py n 30 dihedral+synth > _cv_both.log 2>&1
echo ALLDONE > _cv_done.flag
