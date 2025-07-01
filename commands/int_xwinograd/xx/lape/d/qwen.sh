python intervention_tasks.py \
    --dataset_kaggle "inayarahmanisa/lsnxx-qwen7-flores" \
    --lsn_filename "qwen7_flores" \
    --ld_filename "lang_dict" \
    --hf_token "***REMOVED***" \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
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
    --batch_size 4 \
    > qwen_xwinogradxx.txt 2>&1