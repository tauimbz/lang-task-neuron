python intervention_tasks.py \
    --dataset_kaggle "inayarahmanisa/lsnxx-sea1-5-flores" \
    --lsn_filename "sea1-5_flores" \
    --ld_filename "lang_dict" \
    --dataset_kaggle_replacer "inayarahmanisa/activationxx-sea1-5-neurons" \
    --replacer_filename "median.pt" \
    --model_name "SeaLLMs/SeaLLMs-v3-1.5B-Chat" \
    --dataset_name "CohereLabs/include-lite-44" \
    --split test \
    --replace_method fixed \
    --operation_non_target ".1" \
    --operation_target "=10" \
    --metrics "acc" \
    --kaggle_dataname_to_save "accxx-include-lape-pmed" \
    --parent_dir_to_save "" \
    --selected_langs "Dutch" "Indonesian" "Malay" "Vietnamese" "Japanese" "Chinese" "French" "Portuguese" "Russian" "Estonian" "Italian"  "Tamil" "Turkish" \
    --target_langs 1 2 3 4 5 6 7 8 9 10 12 15 17 \
    --is_update \
    --batch_size 1 \
    > sea_mini_includexx.txt 2>&1