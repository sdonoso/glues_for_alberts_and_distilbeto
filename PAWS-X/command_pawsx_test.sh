python run_glue.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-paws-x/result_albeto_tiny \
  --train_file /data/sedonoso/datasets/PAWS-X/translated_train.json \
  --validation_file /data/sedonoso/datasets/PAWS-X/dev_2k.json \
  --max_seq_length 512 \
  --test_file /data/sedonoso/datasets/PAWS-X/test_2k.json \
  --output_dir /data/sedonoso/memoria/all_results/result-paws-x/result_albeto_tiny \
  --do_predict \
  --use_fast_tokenizer \
  --per_device_train_batch_size 16 \
  --logging_dir /data/sedonoso/memoria/all_results/result-paws-x/result_albeto_tiny \
  --seed 42 \
  --fp16 true \
  --cache_dir /data/sedonoso/cache \
;
python run_glue.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-paws-x/result_albeto_base \
  --train_file /data/sedonoso/datasets/PAWS-X/translated_train.json \
  --validation_file /data/sedonoso/datasets/PAWS-X/dev_2k.json \
  --max_seq_length 512 \
  --test_file /data/sedonoso/datasets/PAWS-X/test_2k.json \
  --output_dir /data/sedonoso/memoria/all_results/result-paws-x/result_albeto_base \
  --do_predict \
  --use_fast_tokenizer \
  --per_device_train_batch_size 16 \
  --logging_dir /data/sedonoso/memoria/all_results/result-paws-x/result_albeto_base \
  --seed 42 \
  --fp16 true \
  --cache_dir /data/sedonoso/cache \
;
python run_glue.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-paws-x/result_albeto_base2 \
  --train_file /data/sedonoso/datasets/PAWS-X/translated_train.json \
  --validation_file /data/sedonoso/datasets/PAWS-X/dev_2k.json \
  --max_seq_length 512 \
  --test_file /data/sedonoso/datasets/PAWS-X/test_2k.json \
  --output_dir /data/sedonoso/memoria/all_results/result-paws-x/result_albeto_base2 \
  --do_predict \
  --use_fast_tokenizer \
  --per_device_train_batch_size 16 \
  --logging_dir /data/sedonoso/memoria/all_results/result-paws-x/result_albeto_base2 \
  --seed 42 \
  --fp16 true \
  --cache_dir /data/sedonoso/cache \
;
python run_glue.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-paws-x/result_albeto_base3 \
  --train_file /data/sedonoso/datasets/PAWS-X/translated_train.json \
  --validation_file /data/sedonoso/datasets/PAWS-X/dev_2k.json \
  --max_seq_length 512 \
  --test_file /data/sedonoso/datasets/PAWS-X/test_2k.json \
  --output_dir /data/sedonoso/memoria/all_results/result-paws-x/result_albeto_base3 \
  --do_predict \
  --use_fast_tokenizer \
  --per_device_train_batch_size 16 \
  --logging_dir /data/sedonoso/memoria/all_results/result-paws-x/result_albeto_base3 \
  --seed 42 \
  --fp16 true \
  --cache_dir /data/sedonoso/cache \
;
python run_glue.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-paws-x/result_albeto_large \
  --train_file /data/sedonoso/datasets/PAWS-X/translated_train.json \
  --validation_file /data/sedonoso/datasets/PAWS-X/dev_2k.json \
  --max_seq_length 512 \
  --test_file /data/sedonoso/datasets/PAWS-X/test_2k.json \
  --output_dir /data/sedonoso/memoria/all_results/result-paws-x/result_albeto_large \
  --do_predict \
  --use_fast_tokenizer \
  --per_device_train_batch_size 16 \
  --logging_dir /data/sedonoso/memoria/all_results/result-paws-x/result_albeto_large \
  --seed 42 \
  --fp16 true \
  --cache_dir /data/sedonoso/cache \
;
python run_glue.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-paws-x/result_albeto_xlarge \
  --train_file /data/sedonoso/datasets/PAWS-X/translated_train.json \
  --validation_file /data/sedonoso/datasets/PAWS-X/dev_2k.json \
  --max_seq_length 512 \
  --test_file /data/sedonoso/datasets/PAWS-X/test_2k.json \
  --output_dir /data/sedonoso/memoria/all_results/result-paws-x/result_albeto_xlarge \
  --do_predict \
  --use_fast_tokenizer \
  --per_device_train_batch_size 16 \
  --logging_dir /data/sedonoso/memoria/all_results/result-paws-x/result_albeto_xlarge \
  --seed 42 \
  --fp16 true \
  --cache_dir /data/sedonoso/cache \
;
python run_glue.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-paws-x/result_albeto_xxlarge \
  --train_file /data/sedonoso/datasets/PAWS-X/translated_train.json \
  --validation_file /data/sedonoso/datasets/PAWS-X/dev_2k.json \
  --max_seq_length 512 \
  --test_file /data/sedonoso/datasets/PAWS-X/test_2k.json \
  --output_dir /data/sedonoso/memoria/all_results/result-paws-x/result_albeto_xxlarge \
  --do_predict \
  --use_fast_tokenizer \
  --per_device_train_batch_size 16 \
  --logging_dir /data/sedonoso/memoria/all_results/result-paws-x/result_albeto_xxlarge \
  --seed 42 \
  --fp16 true \
  --cache_dir /data/sedonoso/cache \
