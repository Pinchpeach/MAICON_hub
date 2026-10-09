# 25에폭(30 미만) x 2시드: 원본 라벨 vs 정리된 라벨(data_fixed 633장)
for s in 0 1; do
python p1_train.py data_local.yaml b_orig25_s$s 640 25 "{\"seed\":$s,\"close_mosaic\":5}" > versions/_p1_b_orig25_s$s.log 2>&1
python p1_train.py data_fixed.yaml b_fix25_s$s 640 25 "{\"seed\":$s,\"close_mosaic\":5}" > versions/_p1_b_fix25_s$s.log 2>&1
done
echo DONE > versions/_p1_run2.flag
