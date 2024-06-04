# export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# python -u run.py \
#   --task_name global_forecast \
#   --is_training 1 \
#   --root_path ./OperStation \
#   --model_id weather_48_24 \
#   --model $model_name \
#   --data Global_Weather_Station \
#   --features M \
#   --seq_len 48 \
#   --label_len 24 \
#   --pred_len 24 \
#   --e_layers 3 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 5 \
#   --dec_in 5 \
#   --c_out 21 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --itr 1 \
#   --num_workers 8 \
#   --target 'TMP' \
#   --train_steps 300000 \
#   --val_steps 20000 \
#   --batch_size 1024 \
#   --patience  3  \
#   --gpu 1 \
#   --lradj cosine_iter \
#   --inverse  &

# python -u run.py \
#   --task_name global_forecast \
#   --is_training 0 \
#   --root_path ./OperStation \
#   --model_id weather_48_72 \
#   --model $model_name \
#   --data Global_Weather_Station \
#   --features M \
#   --seq_len 48 \
#   --label_len 24 \
#   --pred_len 72 \
#   --e_layers 3 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 5 \
#   --dec_in 5 \
#   --c_out 21 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --itr 1 \
#   --num_workers 4 \
#   --target 'TMP' \
#   --train_steps 300000 \
#   --val_steps 20000 \
#   --batch_size 1024 \
#   --patience  3  \
#   --gpu 3 \
#   --lradj cosine_iter \
#   --inverse 


python -u run.py \
  --task_name global_forecast \
  --is_training 0\
  --root_path ./OperStation \
  --model_id weather_48_120 \
  --model $model_name \
  --data Global_Weather_Station \
  --features M \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 120 \
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
  --gpu 3 \
  --lradj cosine_iter \
  --inverse 


python -u run.py \
  --task_name global_forecast \
  --is_training 0 \
  --root_path ./OperStation \
  --model_id weather_48_168 \
  --model $model_name \
  --data Global_Weather_Station \
  --features M \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 168 \
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
  --gpu 4 \
  --lradj cosine_iter \
  --inverse 