accelerate launch run_clm_no_trainer.py \
    --dataset_name wikimedia/wikipedia \
    --dataset_config_name 20231101.en \
    --model_name_or_path gpt2 \
    --output_dir ./tmp/test-clm \
    --per_device_train_batch_size 32 \
    --preprocessing_num_workers 48 \
    #--overwrite_cache \

