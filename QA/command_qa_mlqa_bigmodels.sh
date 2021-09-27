python run_qa.py \
  --model_name_or_path  /data/sedonoso/modelos/albeto_xxlarge \
  --train_file /data/sedonoso/datasets/QA/MLQA/es_squad-translate-train-train-v1.1.json \
  --validation_file /data/sedonoso/datasets/QA/MLQA/es_squad-translate-train-dev-v1.1.json \
  --max_seq_length 384 \
  --output_dir /data/sedonoso/memoria/all_results/result-qa-mlqa/result_albeto_xxlarge \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --learning_rate 3e-6 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --weight_decay 0.01 \
  --num_train_epochs 4.0 \
  --warmup_ratio 0.1 \
  --doc_stride 128 \
  --logging_dir /data/sedonoso/all_results/result-qa/result_albeto_xxlarge \
  --save_strategy epoch \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sedonoso/cache \
;
python run_qa.py \
  --model_name_or_path  /data/sedonoso/modelos/albeto_xlarge \
  --train_file /data/sedonoso/datasets/QA/MLQA/es_squad-translate-train-train-v1.1.json \
  --validation_file /data/sedonoso/datasets/QA/MLQA/es_squad-translate-train-dev-v1.1.json \
  --max_seq_length 384 \
  --output_dir /data/sedonoso/memoria/all_results/result-qa-mlqa/result_albeto_xlarge \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --learning_rate 3e-6 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --weight_decay 0.01 \
  --num_train_epochs 4.0 \
  --warmup_ratio 0.1 \
  --doc_stride 128 \
  --logging_dir /data/sedonoso/all_results/result-qa/result_albeto_xlarge \
  --save_strategy epoch \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sedonoso/cache \
;
python run_qa.py \
  --model_name_or_path  /data/sedonoso/modelos/albeto_large \
  --train_file /data/sedonoso/datasets/QA/MLQA/es_squad-translate-train-train-v1.1.json \
  --validation_file /data/sedonoso/datasets/QA/MLQA/es_squad-translate-train-dev-v1.1.json \
  --max_seq_length 384 \
  --output_dir /data/sedonoso/memoria/all_results/result-qa-mlqa/result_albeto_large \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --learning_rate 3e-6 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --weight_decay 0.01 \
  --num_train_epochs 4.0 \
  --warmup_ratio 0.1 \
  --doc_stride 128 \
  --logging_dir /data/sedonoso/all_results/result-qa/result_albeto_large \
  --save_strategy epoch \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sedonoso/cache