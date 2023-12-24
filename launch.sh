accelerate launch --main_process_port 29550 run_clm_no_trainer.py \
    --dataset_name wikimedia/wikipedia \
    --dataset_config_name 20231101.en \
    --model_name_or_path gpt2 \
    --output_dir ./tmp/test-clm \
    --block_size 1024 \
    --checkpointing_steps 100 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 20 \
    --preprocessing_num_workers 48 \
    --with_tracking \
    --report_to wandb \
    #--overwrite_cache \

