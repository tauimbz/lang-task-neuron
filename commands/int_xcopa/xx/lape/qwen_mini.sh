python intervention_tasks.py \
    --dataset_kaggle "inayarahmanisa/activationxx-qwen05-neurons" \
    --lsn_filename "raw_act_lsn_qwen05.pt" \
    --ld_filename "ld_flores200" \
    --dataset_kaggle_replacer "inayarahmanisa/activationxx-qwen05-neurons" \
    --replacer_filename "max.pt" \
    --hf_token "***REMOVED***" \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --dataset_name "cambridgeltl/xcopa" \
    --split test \
    --replace_method fixed \
    --operation_non_target ".1" \
    --operation_target "=10" \
    --metrics "acc" \
    --kaggle_dataname_to_save "accxx-xcopa-lape" \
    --parent_dir_to_save "" \
    --selected_langs "et" "ht" "id" "it" "qu" "sw" "ta" "th" "tr" "vi" "zh" \
    --target_langs 10 11 2 12 13 14 15 16 17 4 6 \
    --batch_size 6 \
    > qwen_mini_xcopaxx.txt 2>&1