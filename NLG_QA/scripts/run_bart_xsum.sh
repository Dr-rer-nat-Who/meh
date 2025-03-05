# accelerate launch --multi_gpu --num_machine=1 --num_processes=1 --main_process_port=8679 --mixed_precision="no" \
# examples/summarization/run_summarization_no_trainer.py \
# --model_name_or_path facebook/bart-base \
# --dataset_name cnn_dailymail --dataset_config "3.0.0" \
# --apply_lora --apply_elalora --lora_type svd \
# --target_rank 2  --lora_r 2  \
# --b 1 --k 1 \
# --reg_orth_coef 0.1 \
# --init_warmup 3000 --final_warmup 5000 --mask_interval 100 \
# --beta1 0.85 --beta2 0.85 \
# --lora_module q_proj,k_proj,v_proj,out_proj,fc1,fc2 \
# --per_device_train_batch_size 8 --learning_rate 5e-4 \
# --num_train_epochs 5 --num_warmup_steps 3000 \
# --max_source_length 768 --max_target_length 64 --max_length 768 \
# --pad_to_max_length --num_beams 8 \
# --per_device_eval_batch_size 8 \
# --seed 9 \
# --enable_scheduler \
# --with_tracking \
# --tb_writter_loginterval 500 \
# --output_dir ./bart-base/xsum 

python examples/summarization/run_summarization.py \
--model_name_or_path facebook/bart-base \
--dataset_name EdinburghNLP/xsum \
--apply_lora --apply_elalora --lora_type svd \
--target_rank 6  --lora_r 6  --lora_alpha 16 \
--do_train --do_eval \
--b 6 --k 2 \
--reg_orth_coef 0.1 \
--init_warmup 8000 --final_warmup 10000 --mask_interval 200 \
--beta1 0.85 --beta2 0.85 \
--lora_module q_proj,k_proj,v_proj,out_proj,fc1,fc2 \
--per_device_train_batch_size 32 --learning_rate 5e-4 \
--num_train_epochs 8 \
--max_source_length 1024 --max_target_length 160 \
--pad_to_max_length --num_beams 8 \
--per_device_eval_batch_size 32 \
--save_strategy steps --save_steps 1000000 \
--seed 6 \
--enable_scheduler \
--tb_writter_loginterval 50000 \
--predict_with_generate \
--root_output_dir ./ela_bart_base/xsum_8e_r6 \
--overwrite_output_dir 