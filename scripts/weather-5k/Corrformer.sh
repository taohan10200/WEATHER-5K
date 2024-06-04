# export CUDA_VISIBLE_DEVICES=0

model_name=Corrformer
seq_len=48
pred_len_arr=(72 120 168)
gpu_arr=(0 1 2)

for ((i=0; i<${#pred_len_arr[@]}; i++))
do  
    pred_len=${pred_len_arr[i]}
    gpu=${gpu_arr[i]}
    python -u run.py \
      --task_name global_forecast \
      --is_training 1 \
      --root_path ./OperStation \
      --model_id weather_$seq_len'_'$pred_len \
      --model $model_name \
      --data Dataset_Weather_Stations_ALL \
      --features M \
      --seq_len $seq_len  \
      --label_len 24 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 1 \
      --enc_in 80 \
      --dec_in 80 \
      --c_out 80 \
      --des 'Exp' \
      --itr 1 \
      --d_model 768 \
      --n_heads 16 \
      --num_workers 6 \
      --target 'TMP' \
      --train_steps 300000 \
      --val_steps 20000 \
      --batch_size 1 \
      --patience  3  \
      --gpu $gpu \
      --lradj cosine_iter \
      --inverse &
done
