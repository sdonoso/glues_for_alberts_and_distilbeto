python run_ner.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-ner-c/result_albeto_tiny \
  --max_seq_length 512 \
  --output_dir /data/sedonoso/memoria/all_results/result-ner-c/result_albeto_tiny \
  --use_fast_tokenizer  \
  --do_predict \
  --per_device_train_batch_size 32 \
  --logging_dir /data/sedonoso/all_results/result-ner-c/result_albeto_tiny \
  --seed 42 \
  --fp16 true\
  --cache_dir /data/sedonoso/cache \
;
python run_ner.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-ner-c/result_albeto_base \
  --max_seq_length 512 \
  --output_dir /data/sedonoso/memoria/all_results/result-ner-c/result_albeto_base \
  --use_fast_tokenizer  \
  --do_predict \
  --per_device_train_batch_size 32 \
  --logging_dir /data/sedonoso/all_results/result-ner-c/result_albeto_base \
  --seed 42 \
  --fp16 true\
  --cache_dir /data/sedonoso/cache \
;
python run_ner.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-ner-c/result_albeto_base2 \
  --max_seq_length 512 \
  --output_dir /data/sedonoso/memoria/all_results/result-ner-c/result_albeto_base2 \
  --use_fast_tokenizer  \
  --do_predict \
  --per_device_train_batch_size 32 \
  --logging_dir /data/sedonoso/all_results/result-ner-c/result_albeto_base2 \
  --seed 42 \
  --fp16 true\
  --cache_dir /data/sedonoso/cache \
;
python run_ner.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-ner-c/result_albeto_base3 \
  --max_seq_length 512 \
  --output_dir /data/sedonoso/memoria/all_results/result-ner-c/result_albeto_base3 \
  --use_fast_tokenizer  \
  --do_predict \
  --per_device_train_batch_size 32 \
  --logging_dir /data/sedonoso/all_results/result-ner-c/result_albeto_base3 \
  --seed 42 \
  --fp16 true\
  --cache_dir /data/sedonoso/cache \
;
python run_ner.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-ner-c/result_albeto_large \
  --max_seq_length 512 \
  --output_dir /data/sedonoso/memoria/all_results/result-ner-c/result_albeto_large \
  --use_fast_tokenizer  \
  --do_predict \
  --per_device_train_batch_size 32 \
  --logging_dir /data/sedonoso/all_results/result-ner-c/result_albeto_large \
  --seed 42 \
  --fp16 true\
  --cache_dir /data/sedonoso/cache \
;
python run_ner.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-ner-c/result_albeto_xlarge \
  --max_seq_length 512 \
  --output_dir /data/sedonoso/memoria/all_results/result-ner-c/result_albeto_xlarge \
  --use_fast_tokenizer  \
  --do_predict \
  --per_device_train_batch_size 32 \
  --logging_dir /data/sedonoso/all_results/result-ner-c/result_albeto_xlarge \
  --seed 42 \
  --fp16 true\
  --cache_dir /data/sedonoso/cache \
;
python run_ner.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-ner-c/result_albeto_xxlarge \
  --max_seq_length 512 \
  --output_dir /data/sedonoso/memoria/all_results/result-ner-c/result_albeto_xxlarge \
  --use_fast_tokenizer  \
  --do_predict \
  --per_device_train_batch_size 32 \
  --logging_dir /data/sedonoso/all_results/result-ner-c/result_albeto_xxlarge \
  --seed 42 \
  --fp16 true\
  --cache_dir /data/sedonoso/cache