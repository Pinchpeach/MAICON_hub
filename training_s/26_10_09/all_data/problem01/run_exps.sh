python train.py yolov8n.pt ctl640e20 640 20 0 '{"batch":16}' > versions/_train_ctl640e20.log 2>&1
python train.py yolov8n.pt fast640e20 640 20 0 '{"batch":8,"optimizer":"AdamW","lr0":0.002,"warmup_epochs":1,"close_mosaic":5}' > versions/_train_fast640e20.log 2>&1
echo ALLDONE > versions/_exps_done.flag
