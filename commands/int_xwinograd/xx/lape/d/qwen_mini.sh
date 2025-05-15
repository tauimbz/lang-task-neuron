python intervention_tasks.py \
    --dataset_kaggle "inayarahmanisa/lsnxx-qwen05-flores" \
    --lsn_filename "qwen05_flores" \
    --ld_filename "lang_dict" \
    --hf_token "***REMOVED***" \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --dataset_name "Muennighoff/xwinograd" \
    --split test \
    --replace_method percent \
    --operation_non_target ".1" \
    --operation_target "=10" \
    --metrics "acc" \
    --kaggle_dataname_to_save "accxx-xwinograd-lape-d" \
    --parent_dir_to_save "" \
    --target_langs 0 7 5 8 9 6 \
    --is_update \
    --batch_size 8 \
    > qwen_mini_xwinogradxx.txt 2>&1