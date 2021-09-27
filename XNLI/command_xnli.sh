python run_xnli.py \
  --model_name_or_path /data/sedonoso/modelos/albeto_tiny \
  --max_seq_length 512 \
  --output_dir /data/sedonoso/all_results/result-xnli/result_albeto_tiny \
  --use_fast_tokenizer True \
  --language es \
  --train_language es \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 32 \
  --weight_decay 0.01 \
  --num_train_epochs 4.0 \
  --warmup_ratio 0.1 \
  --load_best_model_at_end True \
  --logging_dir /data/sedonoso/all_results/result-xnli/result_albeto_tiny \
  --save_strategy epoch \
  --seed 42 \
  --fp16 true \
  --cache_dir /data/sedonoso/cache \
;
python run_xnli.py \
  --model_name_or_path /data/sedonoso/modelos/albeto_base \
  --max_seq_length 512 \
  --output_dir /data/sedonoso/all_results/result-xnli/result_albeto_base \
  --use_fast_tokenizer True \
  --language es \
  --train_language es \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --learning_rate 3e-5 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --weight_decay 0.01 \
  --num_train_epochs 4.0 \
  --warmup_ratio 0.1 \
  --load_best_model_at_end True \
  --logging_dir /data/sedonoso/all_results/result-xnli/result_albeto_base \
  --save_strategy epoch \
  --seed 42 \
  --fp16 true \
  --cache_dir /data/sedonoso/cache \
;
python run_xnli.py \
  --model_name_or_path /data/sedonoso/modelos/albeto_base2 \
  --max_seq_length 512 \
  --output_dir /data/sedonoso/all_results/result-xnli/result_albeto_base2 \
  --use_fast_tokenizer True \
  --language es \
  --train_language es \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --learning_rate 3e-5 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --weight_decay 0.01 \
  --num_train_epochs 4.0 \
  --warmup_ratio 0.1 \
  --load_best_model_at_end True \
  --logging_dir /data/sedonoso/all_results/result-xnli/result_albeto_base2 \
  --save_strategy epoch \
  --seed 42 \
  --fp16 true \
  --cache_dir /data/sedonoso/cache \
;
python run_xnli.py \
  --model_name_or_path /data/sedonoso/modelos/albeto_base3 \
  --max_seq_length 512 \
  --output_dir /data/sedonoso/all_results/result-xnli/result_albeto_base3 \
  --use_fast_tokenizer True \
  --language es \
  --train_language es \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --learning_rate 3e-5 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --weight_decay 0.01 \
  --num_train_epochs 4.0 \
  --warmup_ratio 0.1 \
  --load_best_model_at_end True \
  --logging_dir /data/sedonoso/all_results/result-xnli/result_albeto_base3 \
  --save_strategy epoch \
  --seed 42 \
  --fp16 true \
  --cache_dir /data/sedonoso/cache



