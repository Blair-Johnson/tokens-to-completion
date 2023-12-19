python -i run_clm_no_trainer.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir /tmp/test-clm \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 32 \
    --overwrite_output_dir \
    --do_train \
    --do_eval 

