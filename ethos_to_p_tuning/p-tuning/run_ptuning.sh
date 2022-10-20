#python -u -m paddle.distributed.launch --gpus "0" \
python ptuning.py \
    --task_name "bustm" \
    --device cpu \
    --p_embedding_num 1 \
    --save_dir "checkpoints" \
    --batch_size 32 \
    --learning_rate 5E-5 \
    --epochs 10 \
    --max_seq_length 512 \
    --rdrop_coef 0 \
