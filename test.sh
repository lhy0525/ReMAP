# store weight in the dir ./weight like mvtec_prompt.pt, mvtec_adaptor.pt which are trained on mvtec dataset
# and visa_prompt.pt, visa_adaptor.pt which are trained on visa dataset
# zero-shot
data_dir=./data
# CUDA_VISIBLE_DEVICES=0  python test.py  --dataset mvtec --weight ./weight/best_maxf1_mvtec.pt --data_dir $data_dir 
# CUDA_VISIBLE_DEVICES=0 python test.py   --dataset visa  --test_dataset mvtec --weight ./weight/best_maxf1_visa.pt  --data_dir $data_dir 

for k in 1 2 4; do
  for i in $(seq 1 ); do  # 需要多跑几次可自行调整
    CUDA_VISIBLE_DEVICES=0 python test.py \
      --dataset mvtec \
      --test_dataset visa \
      --weight ./weight/best_maxf1_mvtec.pt \
      --fewshot $k \
      --alpha 0.1 \
      --data_dir $data_dir \
      --result_dir ./results/few_shot/$k/run_$i
      --vis 1
  done
done

# for k in 1 2 4; do
#   for i in $(seq 1 ); do  # 
#     CUDA_VISIBLE_DEVICES=0 python test.py \
#       --dataset visa \
#       --test_dataset mvtec \
#       --weight ./weight/best_maxf1_visa.pt \
#       --fewshot $k \
#       --alpha 0.1 \
#       --data_dir $data_dir \
#       --result_dir ./results/few_shot/$k/run_$i
#       --vis 1
#   done
# done
