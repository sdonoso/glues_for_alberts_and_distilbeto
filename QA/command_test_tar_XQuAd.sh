python run_qa.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-qa-tar/result_albeto_tiny \
  --do_predict \
  --test_file /data/sedonoso/datasets/QA/Xquad/xquad.es.json \
  --output_dir /data/sedonoso/memoria/all_results/result-qa-tar/result-tar-xquad/result_albeto_tiny \
  --logging_dir /data/sedonoso/memoria/all_results/result-qa-tar/result-tar-xquad/result_albeto_tiny \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sedonoso/cache \
;
python run_qa.py \
  --model_name_or_path /data/sedonoso/memoria/all_results/result-qa-tar/result_albeto_base \
  --do_predict \
  --test_file /data/sedonoso/datasets/QA/Xquad/xquad.es.json \
  --output_dir /data/sedonoso/memoria/all_results/result-qa-tar/result-tar-xquad/result_albeto_base \
  --logging_dir /data/sedonoso/memoria/all_results/result-qa-tar/result-tar-xquad/result_albeto_base \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sedonoso/cache \
;
python run_qa.py \
  --model_name_or_path  /data/sedonoso/memoria/all_results/result-qa-tar/result_albeto_base2 \
  --do_predict \
  --test_file /data/sedonoso/datasets/QA/Xquad/xquad.es.json \
  --output_dir /data/sedonoso/memoria/all_results/result-qa-tar/result-tar-xquad/result_albeto_base2 \
  --logging_dir /data/sedonoso/memoria/all_results/result-qa-tar/result-tar-xquad/result_albeto_base2 \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sedonoso/cache \
;
python run_qa.py \
  --model_name_or_path  /data/sedonoso/memoria/all_results/result-qa-tar/result_albeto_base3 \
  --do_predict \
  --test_file /data/sedonoso/datasets/QA/Xquad/xquad.es.json \
  --output_dir /data/sedonoso/memoria/all_results/result-qa-tar/result-tar-xquad/result_albeto_base3 \
  --logging_dir /data/sedonoso/memoria/all_results/result-qa-tar/result-tar-xquad/result_albeto_base3 \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sedonoso/cache \
;
python run_qa.py \
  --model_name_or_path  /data/sedonoso/memoria/all_results/result-qa-tar/result_albeto_large \
  --do_predict \
  --test_file /data/sedonoso/datasets/QA/Xquad/xquad.es.json \
  --output_dir /data/sedonoso/memoria/all_results/result-qa-tar/result-tar-xquad/result_albeto_large \
  --logging_dir /data/sedonoso/memoria/all_results/result-qa-tar/result-tar-xquad/result_albeto_large \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sedonoso/cache \
;
python run_qa.py \
  --model_name_or_path  /data/sedonoso/memoria/all_results/result-qa-tar/result_albeto_xlarge \
  --do_predict \
  --test_file /data/sedonoso/datasets/QA/Xquad/xquad.es.json \
  --output_dir /data/sedonoso/memoria/all_results/result-qa-tar/result-tar-xquad/result_albeto_xlarge \
  --logging_dir /data/sedonoso/memoria/all_results/result-qa-tar/result-tar-xquad/result_albeto_xlarge \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sedonoso/cache \
;
python run_qa.py \
  --model_name_or_path  /data/sedonoso/memoria/all_results/result-qa-tar/result_albeto_xxlarge \
  --do_predict \
  --test_file /data/sedonoso/datasets/QA/Xquad/xquad.es.json \
  --output_dir /data/sedonoso/memoria/all_results/result-qa-tar/result-tar-xquad/result_albeto_xxlarge \
  --logging_dir /data/sedonoso/memoria/all_results/result-qa-tar/result-tar-xquad/result_albeto_xxlarge \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sedonoso/cache