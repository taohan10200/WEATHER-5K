#!/bin/bash

model_name=Mamba
seq_len=48

pred_len_arr=(72 )
gpu_arr=(0 )

for ((i=0; i<${#pred_len_arr[@]}; i++))
do
  pred_len=${pred_len_arr[i]}
  gpu=${gpu_arr[i]}

  python -u run.py \
    --task_name global_forecast \
    --is_training 1 \
    --root_path ./WEATHER-5K \
    --model_id weather_$seq_len'_'$pred_len \
    --model $model_name \
    --data Global_Weather_Station \
    --features M \
    --seq_len $pred_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 5 \
    --d_ff 16 \
    --c_out 5 \
    --d_model 128 \
    --des 'Exp' \
    --itr 1 \
    --num_workers 4 \
    --target 'TMP' \
    --train_steps 300000 \
    --val_steps 300000 \
    --batch_size 1024 \
    --patience 3 \
    --gpu $gpu \
    --lradj cosine_iter \
    --inverse  
done