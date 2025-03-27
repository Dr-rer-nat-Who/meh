#!/bin/bash
#SBATCH --job-name=elalora_specialized
#SBATCH -c 2
#SBATCH -n 1
#SBATCH -t 0-2:30             # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu                  # Partition to submit to
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1      
#SBATCH --mem=64GB

#SBATCH -o ~/slurm_output/myoutput_%j.out      # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ~/slurm_output/myerrors_%j.err      # File to which STDERR will be written, %j inserts jobid

#SBATCH --mail-type=ALL
#SBATCH --mail-user=huandongchang@fas.harvard.edu

# --- load env here ---
module load python/3.10.12-fasrc01
source activate
source activate IMAGE
# ---------------------


python -c 'print("Hi Huandong. Your job is running!")'

cd /n/home04/huandongchang/ElaLoRA/IMAGE_CLASS
# --- run your code here ---
python image-classification/run_image_classification.py \
    --dataset_name /n/netscratch/kung_lab/Everyone/zma/vtab-1k/eurosat/ \
    --remove_unused_columns False \
    --label_column_name label \
    --do_train \
    --do_eval \
    --apply_lora --apply_elalora --lora_type svd \
    --target_rank 8  --lora_r 8  --lora_alpha 16 \
    --b 4 --k 2 \
    --init_warmup 500 --final_warmup 1000 --mask_interval 200 \
    --beta1 0.85 --beta2 0.85 \
    --lora_module query,key,value,intermediate,layer.output \
    --enable_scheduler \
    --tb_writter_loginterval 50000 \
    --learning_rate 1e-3 \
    --num_train_epochs 100 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 64 \
    --logging_steps 1000 \
    --evaluation_strategy steps --eval_steps 1000 \
    --save_total_limit 1 \
    --seed 6 \
    --root_output_dir ./ela/eurosat_r8_100e \
    --overwrite_output_dir

python image-classification/run_image_classification.py \
    --dataset_name /n/netscratch/kung_lab/Everyone/zma/vtab-1k/resisc45/ \
    --remove_unused_columns False \
    --label_column_name label \
    --do_train \
    --do_eval \
    --apply_lora --apply_elalora --lora_type svd \
    --target_rank 8  --lora_r 8  --lora_alpha 16 \
    --b 4 --k 2 \
    --init_warmup 500 --final_warmup 1000 --mask_interval 200 \
    --beta1 0.85 --beta2 0.85 \
    --lora_module query,key,value,intermediate,layer.output \
    --enable_scheduler \
    --tb_writter_loginterval 50000 \
    --learning_rate 1e-3 \
    --num_train_epochs 100 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 64 \
    --logging_steps 1000 \
    --evaluation_strategy steps --eval_steps 1000 \
    --save_total_limit 1 \
    --seed 6 \
    --root_output_dir ./ela/resisc45_r8_100e \
    --overwrite_output_dir


python image-classification/run_image_classification.py \
    --dataset_name /n/netscratch/kung_lab/Everyone/zma/vtab-1k/diabetic_retinopathy/ \
    --remove_unused_columns False \
    --label_column_name label \
    --do_train \
    --do_eval \
    --apply_lora --apply_elalora --lora_type svd \
    --target_rank 8  --lora_r 8  --lora_alpha 16 \
    --b 4 --k 2 \
    --init_warmup 500 --final_warmup 1000 --mask_interval 200 \
    --beta1 0.85 --beta2 0.85 \
    --lora_module query,key,value,intermediate,layer.output \
    --enable_scheduler \
    --tb_writter_loginterval 50000 \
    --learning_rate 1e-3 \
    --num_train_epochs 100 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 64 \
    --logging_steps 1000 \
    --evaluation_strategy steps --eval_steps 1000 \
    --save_total_limit 1 \
    --seed 6 \
    --root_output_dir ./ela/diabetic_r8_100e \
    --overwrite_output_dir

python image-classification/run_image_classification.py \
    --dataset_name /n/netscratch/kung_lab/Everyone/zma/vtab-1k/clevr_dist/ \
    --remove_unused_columns False \
    --label_column_name label \
    --do_train \
    --do_eval \
    --apply_lora --apply_elalora --lora_type svd \
    --target_rank 8  --lora_r 8  --lora_alpha 16 \
    --b 4 --k 2 \
    --init_warmup 500 --final_warmup 1000 --mask_interval 200 \
    --beta1 0.85 --beta2 0.85 \
    --lora_module query,key,value,intermediate,layer.output \
    --enable_scheduler \
    --tb_writter_loginterval 50000 \
    --learning_rate 1e-3 \
    --num_train_epochs 100 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 64 \
    --logging_steps 1000 \
    --evaluation_strategy steps --eval_steps 1000 \
    --save_total_limit 1 \
    --seed 6 \
    --root_output_dir ./ela/clevr_dist_r8_100e \
    --overwrite_output_dir


python image-classification/run_image_classification.py \
    --dataset_name /n/netscratch/kung_lab/Everyone/zma/vtab-1k/clevr_count/ \
    --remove_unused_columns False \
    --label_column_name label \
    --do_train \
    --do_eval \
    --apply_lora --apply_elalora --lora_type svd \
    --target_rank 8  --lora_r 8  --lora_alpha 16 \
    --b 4 --k 2 \
    --init_warmup 500 --final_warmup 1000 --mask_interval 200 \
    --beta1 0.85 --beta2 0.85 \
    --lora_module query,key,value,intermediate,layer.output \
    --enable_scheduler \
    --tb_writter_loginterval 50000 \
    --learning_rate 1e-3 \
    --num_train_epochs 100 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 64 \
    --logging_steps 1000 \
    --evaluation_strategy steps --eval_steps 1000 \
    --save_total_limit 1 \
    --seed 6 \
    --root_output_dir ./ela/clevr_count_r8_100e \
    --overwrite_output_dir

# --------------------------

python -c 'print("Hi Huandong. Everything is done!")'
