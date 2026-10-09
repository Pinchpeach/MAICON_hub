# 1) ⑤+④: 소수 클래스 이미지 3회 반복 + 분류 손실 가중(cls 1.0) + 축소 증강 완화(scale 0.2)
python o_train.py data_o_bal.yaml o_n640bal 640 45 '{"cls":1.0,"scale":0.2}' > versions/_train_o_n640bal.log 2>&1
echo DONE1 > versions/_o2_step1.flag
# 2) ①: 입력 960
python o_train.py data_local.yaml o_n960 960 60 '{"batch":8}' > versions/_train_o_n960.log 2>&1
echo ALLDONE > versions/_o2_done.flag
