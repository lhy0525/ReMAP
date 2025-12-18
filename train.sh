
# train for zero-shot model
data_dir=./data
# CUDA_VISIBLE_DEVICES=0 python main.py --log_dir ./train_log --dataset mvtec --test_dataset visa --data_dir $data_dir  --epoch 40    --best_ds visa
CUDA_VISIBLE_DEVICES=0 python main.py --log_dir ./train_log --dataset visa --test_dataset mvtec --data_dir $data_dir  --epoch 40   --best_ds mvtec




