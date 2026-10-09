# 정리 데이터(data_fixed) 기반 학습 개선 후보, 25에폭 x 2시드. 기준은 b_fix25_s0/s1 (640, 기본 설정)
for s in 0 1; do
python p1_train.py data_fixed.yaml c_sz800_s$s 800 25 "{\"seed\":$s,\"close_mosaic\":5}" > versions/_p1_c_sz800_s$s.log 2>&1
python p1_train.py data_fixed.yaml c_cls15_s$s 640 25 "{\"seed\":$s,\"close_mosaic\":5,\"cls\":1.5}" > versions/_p1_c_cls15_s$s.log 2>&1
python p1_train.py data_fixed.yaml c_aug_s$s 640 25 "{\"seed\":$s,\"close_mosaic\":5,\"mixup\":0.15,\"copy_paste\":0.1,\"degrees\":5,\"flipud\":0.1}" > versions/_p1_c_aug_s$s.log 2>&1
done
echo DONE > versions/_p1_run3.flag
