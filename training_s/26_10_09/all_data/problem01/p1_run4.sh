# 입력 900, 나머지 조건은 b_fix25 와 동일(25에폭, close_mosaic 5, batch 16). 더미 포함 2시드 + 더미 없는 통제 1시드
for s in 0 1; do
python p1_train.py data_dummy.yaml d_sz900dummy_s$s 900 25 "{\"seed\":$s,\"close_mosaic\":5}" > versions/_p1_d_sz900dummy_s$s.log 2>&1
done
python p1_train.py data_fixed.yaml d_sz900only_s0 900 25 "{\"seed\":0,\"close_mosaic\":5}" > versions/_p1_d_sz900only_s0.log 2>&1
echo DONE > versions/_p1_run4.flag
