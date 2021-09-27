python run_ner.py \
  --model_name_or_path /media/sdonoso/sd_flashdri/all_results/result-ner-c/result_albeto_tiny \
  --max_seq_length 128 \
  --output_dir /home/sdonoso/projects/memoria/inference_time/result-inference-time/result_2/result_albeto_tiny \
  --use_fast_tokenizer  \
  --max_test_samples 100 \
  --do_predict \
  --do_lower_case True \
  --per_device_eval_batch_size 1 \
  --seed 42 \
  --cache_dir /home/sdonoso/projects/memoria/inference_time/cache \
;
python run_ner.py \
  --model_name_or_path /media/sdonoso/sd_flashdri/all_results/result-ner-c/result_albeto_base \
  --max_seq_length 128 \
  --output_dir /home/sdonoso/projects/memoria/inference_time/result-inference-time/result_2/result_albeto_base \
  --use_fast_tokenizer  \
  --max_test_samples  100 \
  --do_predict \
  --do_lower_case True \
  --per_device_eval_batch_size 1 \
  --seed 42 \
  --cache_dir /home/sdonoso/projects/memoria/inference_time/cache \
;

python run_ner.py \
  --model_name_or_path /media/sdonoso/sd_flashdri/all_results/result-ner-c/result_albeto_large \
  --max_seq_length 128 \
  --output_dir /home/sdonoso/projects/memoria/inference_time/result-inference-time/result_2/result_albeto_large \
  --use_fast_tokenizer  \
  --max_test_samples 100 \
  --do_predict \
  --do_lower_case True \
  --per_device_eval_batch_size 1 \
  --seed 42 \
  --cache_dir /home/sdonoso/projects/memoria/inference_time/cachee \
;
python run_ner.py \
  --model_name_or_path /media/sdonoso/sd_flashdri/all_results/result-ner-c/result_albeto_xlarge \
  --max_seq_length 128 \
  --output_dir /home/sdonoso/projects/memoria/inference_time/result-inference-time/result_2/result_albeto_xlarge \
  --use_fast_tokenizer  \
  --max_test_samples 100 \
  --do_predict \
  --do_lower_case True \
  --per_device_eval_batch_size 1 \
  --seed 42 \
  --cache_dir /home/sdonoso/projects/memoria/inference_time/cache \
;
python run_ner.py \
  --model_name_or_path /media/sdonoso/sd_flashdri/all_results/result-ner-c/result_albeto_xxlarge \
  --max_seq_length 128 \
  --output_dir /home/sdonoso/projects/memoria/inference_time/result-inference-time/result_2/result_albeto_xxlarge \
  --use_fast_tokenizer  \
  --max_test_samples 100 \
  --do_predict \
  --do_lower_case True \
  --per_device_eval_batch_size 1 \
  --seed 42 \
  --cache_dir /home/sdonoso/projects/memoria/inference_time/cache \
;
python run_ner.py \
  --model_name_or_path /media/sdonoso/sd_flashdri/all_results/result-ner-c/result_distill_sbert \
  --max_seq_length 128 \
  --output_dir /home/sdonoso/projects/memoria/inference_time/result-inference-time/result_2/result_distill_sbert \
  --use_fast_tokenizer  \
  --max_test_samples 100 \
  --do_predict \
  --do_lower_case True \
  --per_device_eval_batch_size 1 \
  --seed 42 \
  --cache_dir /home/sdonoso/projects/memoria/inference_time/cache
