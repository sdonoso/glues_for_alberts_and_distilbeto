python run_qa.py \
  --model_name_or_path  /data/sedonoso/modelos/albeto_tiny \
  --train_file /data/sedonoso/datasets/QA/MLQA/es_squad-translate-train-train-v1.1.json \
  --validation_file /data/sedonoso/datasets/QA/MLQA/es_squad-translate-train-dev-v1.1.json \
  --max_seq_length 384 \
  --output_dir /data/sedonoso/all_results/result-qa-mlqa/result_albeto_tiny \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --learning_rate 3e-5 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --weight_decay 0.01 \
  --num_train_epochs 4.0 \
  --warmup_ratio 0.1 \
  --doc_stride 128 \
  --logging_dir /data/sedonoso/all_results/result-qa-mlqa/result_albeto_tiny \
  --save_strategy epoch \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sedonoso/cache \
;
python run_qa.py \
  --model_name_or_path  /data/sedonoso/modelos/albeto_base \
  --train_file /data/sedonoso/datasets/QA/MLQA/es_squad-translate-train-train-v1.1.json \
  --validation_file /data/sedonoso/datasets/QA/MLQA/es_squad-translate-train-dev-v1.1.json \
  --max_seq_length 384 \
  --output_dir /data/sedonoso/all_results/result-qa-mlqa/result_albeto_base \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --learning_rate 3e-5 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --weight_decay 0.01 \
  --num_train_epochs 4.0 \
  --warmup_ratio 0.1 \
  --doc_stride 128 \
  --logging_dir /data/sedonoso/all_results/result-qa-mlqa/result_albeto_base \
  --save_strategy epoch \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sedonoso/cache \
;
python run_qa.py \
  --model_name_or_path  /data/sedonoso/modelos/albeto_base2 \
  --train_file /data/sedonoso/datasets/QA/MLQA/es_squad-translate-train-train-v1.1.json \
  --validation_file /data/sedonoso/datasets/QA/MLQA/es_squad-translate-train-dev-v1.1.json \
  --max_seq_length 384 \
  --output_dir /data/sedonoso/all_results/result-qa-mlqa/result_albeto_base2 \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --learning_rate 3e-5 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --weight_decay 0.01 \
  --num_train_epochs 4.0 \
  --warmup_ratio 0.1 \
  --doc_stride 128 \
  --logging_dir /data/sedonoso/all_results/result-qa-mlqa/result_albeto_base2 \
  --save_strategy epoch \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sedonoso/cache \
;
python run_qa.py \
  --model_name_or_path  /data/sedonoso/modelos/albeto_base3 \
  --train_file /data/sedonoso/datasets/QA/MLQA/es_squad-translate-train-train-v1.1.json \
  --validation_file /data/sedonoso/datasets/QA/MLQA/es_squad-translate-train-dev-v1.1.json \
  --max_seq_length 384 \
  --output_dir /data/sedonoso/all_results/result-qa-mlqa/result_albeto_base3 \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --learning_rate 3e-5 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --weight_decay 0.01 \
  --num_train_epochs 4.0 \
  --warmup_ratio 0.1 \
  --doc_stride 128 \
  --logging_dir /data/sedonoso/all_results/result-qa-mlqa/result_albeto_base3 \
  --save_strategy epoch \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sedonoso/cache