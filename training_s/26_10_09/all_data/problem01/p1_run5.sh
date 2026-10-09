# 4종을 '차량' 1클래스로 합친 yolov8n (대수 세기 전용). 조건은 b_fix25 와 동일(정리 데이터, 640, 25에폭)
for s in 0 1; do
python p1_train.py data_fixed.yaml e_1cls_s$s 640 25 "{\"seed\":$s,\"close_mosaic\":5,\"single_cls\":true}" > versions/_p1_e_1cls_s$s.log 2>&1
done
echo DONE > versions/_p1_run5.flag
