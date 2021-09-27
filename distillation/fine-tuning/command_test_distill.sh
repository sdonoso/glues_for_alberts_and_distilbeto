python run_xnli.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-xnli/result_distill_sbert \
  --max_seq_length 512 \
  --output_dir /data/sedonoso/memoria/all_results/result-xnli/result_distill_sbert  \
  --use_fast_tokenizer True \
  --language es \
  --train_language es \
  --do_predict \
  --per_device_train_batch_size 32 \
  --logging_dir /data/sedonoso/all_results/result-xnli/result_distill_sbert  \
  --seed 42 \
  --fp16 \
  --do_lower_case True\
  --cache_dir /data/sedonoso/cache \
;
python run_pos.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-pos/result_distill_sbert \
  --do_predict \
  --max_seq_length 512 \
  --output_dir /data/sedonoso/memoria/all_results/result-pos/result_distill_sbert \
  --logging_dir /data/sedonoso/memoria/all_results/result-pos/result_distill_sbert \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sedonoso/cache \
;
python run_glue.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-paws-x/result_distill_sbert \
  --train_file /data/sedonoso/datasets/PAWS-X/translated_train.json \
  --validation_file /data/sedonoso/datasets/PAWS-X/dev_2k.json \
  --max_seq_length 512 \
  --test_file /data/sedonoso/datasets/PAWS-X/test_2k.json \
  --output_dir /data/sedonoso/memoria/all_results/result-paws-x/result_distill_sbert \
  --do_predict \
  --use_fast_tokenizer \
  --per_device_train_batch_size 16 \
  --logging_dir /data/sedonoso/memoria/all_results/result-paws-x/result_distill_sbert \
  --seed 42 \
  --fp16 true \
  --cache_dir /data/sedonoso/cache \
;
python run_ner.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-ner-c/result_distill_sbert \
  --max_seq_length 512 \
  --output_dir /data/sedonoso/memoria/all_results/result-ner-c/result_distill_sbert \
  --use_fast_tokenizer  \
  --do_predict \
  --per_device_train_batch_size 32 \
  --logging_dir /data/sedonoso/all_results/result-ner-c/result_distill_sbert \
  --seed 42 \
  --fp16 true\
  --do_lower_case True\
  --cache_dir /data/sedonoso/cache \
;
python run_qa.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-qa-mlqa/result_distill_sbert \
  --do_predict \
  --test_file /data/sedonoso/datasets/QA/MLQA/test-context-es-question-es.json \
  --output_dir /data/sedonoso/memoria/all_results/result-qa-mlqa/result_distill_sbert \
  --logging_dir /data/sedonoso/memoria/all_results/result-qa-mlqa/result_distill_sbert \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sedonoso/cache \
;
python run_qa.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-qa-tar/result_distill_sbert \
  --do_predict \
  --test_file /data/sedonoso/datasets/QA/Xquad/xquad.es.json \
  --output_dir /data/sedonoso/memoria/all_results/result-qa-tar/result-tar-xquad/result_distill_sbert \
  --logging_dir /data/sedonoso/memoria/all_results/result-qa-tar/result-tar-xquad/result_distill_sbert \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sedonoso/cache \
;
python run_qa.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-qa-tar/result_distill_sbert \
  --do_predict \
  --test_file /data/sedonoso/datasets/QA/MLQA/test-context-es-question-es.json \
  --output_dir /data/sedonoso/memoria/all_results/result-qa-tar/result-tar-mlqa/result_distill_sbert \
  --logging_dir /data/sedonoso/memoria/all_results/result-qa-tar/result-tar-mlqa/result_distill_sbert \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sedonoso/cache \
;
python test_mldoc.py \
  --model-dir /data/sedonoso/memoria/all_results/result-mldoc/result_distill_sbert \
  --do-lower-case \
  --data-dir /data/sedonoso/datasets/MLDoC \
  --output-dir /data/sedonoso/memoria/all_results/result-mldoc/result_distill_sbert
