export NODE_RANK=0
export N_NODES=1

export N_GPU_NODE=2
export WORLD_SIZE=2

pkill -f 'python -u train.py'

python -m torch.distributed.launch \
    --nproc_per_node=$N_GPU_NODE \
    --nnodes=$N_NODES \
    --node_rank $NODE_RANK \
    train.py \
        --force \
        --n_gpu $WORLD_SIZE \
        --student_type distilbert \
        --student_config /data/sedonoso/memoria/distillation/training_configs/distillbert-base-uncased.json \
        --student_pretrained_weights /data/sedonoso/memoria/distillation/serialization_dir/tf_bert-base-uncased_0247911.pth \
        --teacher_type bert \
        --teacher_name /home/sedonoso/bert-models/pytorch \
        --alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 1.0 --alpha_clm 0.0 --mlm \
        --dump_path /home/sedonoso/results_distilbert \
        --gradient_accumulation_steps 100 \
        --batch_size 20 \
        --data_file /home/sedonoso/data/binarized_text.bert-base-uncased.pickle \
        --token_counts /home/sedonoso/data/token_counts.bert-base-uncased.pickle \
        --seed 42