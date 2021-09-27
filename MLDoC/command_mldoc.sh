python finetune_mldoc.py \
  --model-dir /data/sedonoso/modelos/albeto_base \
  --data-dir /data/sedonoso/datasets/MLDoC \
  --output-dir /data/sedonoso/memoria/all_results/result-mldoc/result_albeto_base \
  --learn-rate 3e-5 \
  --batch-size 16 \
  --epochs 3 \
  --max-seq-len 512 \
  --weight-decay 0.01 \
  --warmup 0.1 \
  --seed 42 \
;
python finetune_mldoc.py \
  --model-dir /data/sedonoso/modelos/albeto_base2 \
  --data-dir /data/sedonoso/datasets/MLDoC \
  --output-dir /data/sedonoso/memoria/all_results/result-mldoc/result_albeto_base2 \
  --learn-rate 3e-5 \
  --batch-size 16 \
  --epochs 3 \
  --max-seq-len 512 \
  --weight-decay 0.01 \
  --warmup 0.1 \
  --seed 42 \
;
python finetune_mldoc.py \
  --model-dir /data/sedonoso/modelos/albeto_base3 \
  --data-dir /data/sedonoso/datasets/MLDoC \
  --output-dir /data/sedonoso/memoria/all_results/result-mldoc/result_albeto_base3 \
  --learn-rate 3e-5 \
  --batch-size 16 \
  --epochs 3 \
  --max-seq-len 512 \
  --weight-decay 0.01 \
  --warmup 0.1 \
  --seed 42