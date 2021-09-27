python run_ner.py \
  --model_name_or_path /data/sedonoso/modelos/albeto_xxlarge \
  --max_seq_length 512 \
  --output_dir /data/sedonoso/memoria/all_results/result-ner-c/result_albeto_xxlarge \
  --use_fast_tokenizer \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --learning_rate 3e-5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --weight_decay 0.01 \
  --num_train_epochs 4.0 \
  --warmup_ratio 0.1 \
  --load_best_model_at_end \
  --logging_dir /data/sedonoso/all_results/result-ner-c/result_albeto_xxlarge \
  --save_strategy epoch \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sedonoso/cache \
;
python run_ner.py \
  --model_name_or_path /data/sedonoso/modelos/albeto_xlarge \
  --max_seq_length 512 \
  --output_dir /data/sedonoso/memoria/all_results/result-ner-c/result_albeto_xlarge \
  --use_fast_tokenizer \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --weight_decay 0.01 \
  --num_train_epochs 4.0 \
  --warmup_ratio 0.1 \
  --load_best_model_at_end \
  --logging_dir /data/sedonoso/memoria/all_results/result-ner-c/result_albeto_xlarge \
  --save_strategy epoch \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sedonoso/cache \
;
python run_ner.py \
  --model_name_or_path /data/sedonoso/modelos/albeto_large \
  --max_seq_length 512 \
  --output_dir /data/sedonoso/memoria/all_results/result-ner-c/result_albeto_large \
  --use_fast_tokenizer \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --learning_rate 3e-5 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --weight_decay 0.01 \
  --num_train_epochs 4.0 \
  --warmup_ratio 0.1 \
  --load_best_model_at_end True \
  --logging_dir /data/sedonoso/memoria/all_results/result-ner-c/result_albeto_large \
  --save_strategy epoch \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sedonoso/cache


