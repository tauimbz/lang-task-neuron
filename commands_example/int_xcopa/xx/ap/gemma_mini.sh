python intervention_tasks.py \
    --dataset_kaggle "inayarahmanisa/lsnxx-gemma2-flores" \
    --lsn_filename "map_t354_gemma2_flores.pt" \
    --ld_filename "lang_dict" \
    --dataset_kaggle_replacer "inayarahmanisa/activationxx-gemma2-neurons" \
    --replacer_filename "max.pt" \
    --model_name "google/gemma-2-2b-it" \
    --dataset_name "cambridgeltl/xcopa" \
    --split test \
    --replace_method fixed \
    --operation_non_target ".1" \
    --operation_target "=10" \
    --metrics "acc" \
    --kaggle_dataname_to_save "accxx-xcopa-ap" \
    --parent_dir_to_save "" \
    --selected_langs "et" "ht" "id" "it" "qu" "sw" "ta" "th" "tr" "vi" "zh" \
    --target_langs 10 11 2 12 13 14 15 16 17 4 6 \
    --is_update \
    --batch_size 6 \
    > gemma_mini_xcopaxx.txt 2>&1
