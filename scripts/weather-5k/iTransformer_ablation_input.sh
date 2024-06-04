#!/bin/bash
model_name=iTransformer
pred_len=72
seq_len_arr=(24  96 120)
gpu_arr=(0 1 2 )

for ((i=0; i<${#seq_len_arr[@]}; i++))
do  
    seq_len=${seq_len_arr[i]}
    gpu=${gpu_arr[i]}

    python -u run.py \
      --task_name global_forecast \
      --is_training 0 \
      --root_path ./OperStation \
      --model_id weather_$seq_len'_'$pred_len \
      --model $model_name \
      --data Global_Weather_Station \
      --features M \
      --seq_len $seq_len \
      --label_len 24 \
      --pred_len $pred_len \
      --e_layers 3 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 5 \
      --dec_in 5 \
      --c_out 5 \
      --des 'Exp' \
      --d_model 512 \
      --d_ff 512 \
      --itr 1 \
      --num_workers 4 \
      --target 'TMP' \
      --train_steps 300000 \
      --val_steps 300000 \
      --batch_size 1024 \
      --patience  3  \
      --gpu $gpu \
      --lradj cosine_iter \
      --inverse  &
done