python p1_train.py data_local.yaml a_orig640 640 100 > versions/_p1_a_orig640.log 2>&1
python p1_train.py data_fixed.yaml a_fix640 640 100 > versions/_p1_a_fix640.log 2>&1
echo DONE > versions/_p1_run1.flag
