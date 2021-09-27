#python finetune_mldoc.py \
#  --model-dir /data/sedonoso/modelos/albeto_xxlarge \
#  --data-dir /data/sedonoso/datasets/MLDoC \
#  --output-dir /data/sedonoso/memoria/all_results/result-mldoc/result_albeto_xxlarge \
#  --learn-rate 3e-6 \
#  --batch-size 4 \
#  --epochs 3 \
#  --max-seq-len 512 \
#  --weight-decay 0.01 \
#  --warmup 0.1 \
#  --seed 42 \
#  --overwrite-output-dir
#;
python finetune_mldoc.py \
  --model-dir /data/sedonoso/modelos/albeto_xlarge \
  --data-dir /data/sedonoso/datasets/MLDoC \
  --output-dir /data/sedonoso/memoria/all_results/result-mldoc/result_albeto_xlarge \
  --learn-rate 3e-6 \
  --batch-size 4 \
  --epochs 3 \
  --max-seq-len 512 \
  --weight-decay 0.01 \
  --warmup 0.1 \
  --seed 42 \
  --overwrite-output-dir \
#;
#python finetune_mldoc.py \
#  --model-dir /data/sedonoso/modelos/albeto_large \
#  --data-dir /data/sedonoso/datasets/MLDoC \
#  --output-dir /data/sedonoso/memoria/all_results/result-mldoc/result_albeto_large \
#  --learn-rate 3e-6 \
#  --batch-size 8 \
#  --epochs 3 \
#  --max-seq-len 512 \
#  --weight-decay 0.01 \
#  --warmup 0.1 \
#  --seed 42 \
#  --overwrite-output-dir
