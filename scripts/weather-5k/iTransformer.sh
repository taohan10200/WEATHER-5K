#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
model_name=iTransformer
seq_len=48

pred_len_arr=(120 )
gpu_arr=(0  )

for ((i=0; i<${#pred_len_arr[@]}; i++))
do
  pred_len=${pred_len_arr[i]}
  gpu=${gpu_arr[i]}

  python -u run.py \
  --task_name global_forecast \
  --is_training 0\
  --root_path ./WEATHER-5K \
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
  --c_out 21 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --num_workers 8 \
  --target 'TMP' \
  --train_steps 300000 \
  --val_steps 20000 \
  --batch_size 1024 \
  --patience  3  \
  --gpu $gpu \
  --lradj cosine_iter \
  --inverse 

done
