python3 intervention_tasks.py \
    --dataset_kaggle "inayarahmanisa/lsnxx-gemma9-flores" \
    --lsn_filename "raw_act_lsn_gemma9.pt" \
    --ld_filename "lang_dict" \
    --dataset_kaggle_replacer "inayarahmanisa/activationxx-gemma9-flores" \
    --replacer_filename "max.pt" \
    --model_name "google/gemma-2-9b-it" \
    --dataset_name "cambridgeltl/xcopa" \
    --split test \
    --replace_method fixed \
    --operation_non_target ".1" \
    --operation_target "=10" \
    --metrics "acc" \
    --kaggle_dataname_to_save "accxx-xcopa-raw" \
    --parent_dir_to_save "" \
    --selected_langs "et" "ht" "id" "it" "qu" "sw" "ta" "th" "tr" "vi" "zh" \
    --target_langs 10 11 2 12 13 14 15 16 17 4 6 \
    --is_update \
    --batch_size 4 \
    > gemma_xcopaxx.txt 2>&1
