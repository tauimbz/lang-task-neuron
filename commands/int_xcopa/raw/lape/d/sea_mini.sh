python intervention_tasks.py \
    --dataset_kaggle "inayarahmanisa/lsnxx-sea1-5-flores" \
    --lsn_filename "sea1-5_flores" \
    --ld_filename "lang_dict" \
    --hf_token "***REMOVED***" \
    --model_name "SeaLLMs/SeaLLMs-v3-1.5B-Chat" \
    --dataset_name "cambridgeltl/xcopa" \
    --split test \
    --replace_method percent \
    --operation_non_target ".1" \
    --operation_target "=10" \
    --metrics "acc" \
    --kaggle_dataname_to_save "accxx-xcopa-lape-d" \
    --parent_dir_to_save "" \
    --selected_langs "et" "ht" "id" "it" "qu" "sw" "ta" "th" "tr" "vi" "zh" \
    --target_langs 10 11 2 12 13 14 15 16 17 4 6 \
    --is_update \
    --batch_size 6 \
    > sea_mini_xcopaxx.txt 2>&1