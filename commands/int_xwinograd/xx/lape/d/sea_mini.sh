python intervention_dod.py \
    --dataset_kaggle "inayarahmanisa/lsnxx-sea1-5-flores" \
    --lsn_filename "sea1-5_flores" \
    --ld_filename "lang_dict" \
    --hf_token "***REMOVED***" \
    --model_name "SeaLLMs/SeaLLMs-v3-1.5B-Chat" \
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
    > sea_mini_xwinogradxx.txt 2>&1