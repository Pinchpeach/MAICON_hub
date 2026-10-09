# 이 세션의 실험 (이름 접두 o_). GPU 로 순서대로 실행
python train.py yolov8n.pt o_n640frz 640 60 0 '{"batch":16,"freeze":10,"patience":15}' > versions/_train_o_n640frz.log 2>&1
python train.py yolov8n.pt o_n960 960 60 0 '{"batch":8,"patience":15}' > versions/_train_o_n960.log 2>&1
echo ALLDONE > versions/_o_done.flag
