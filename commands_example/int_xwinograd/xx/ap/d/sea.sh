python intervention_dod.py \
    --dataset_kaggle "inayarahmanisa/lsnxx-sea7-flores" \
    --lsn_filename "map_tb1426_sea7_flores.pt" \
    --ld_filename "lang_dict" \
    --model_name "SeaLLMs/SeaLLMs-v3-7B-Chat" \
    --dataset_name "Muennighoff/xwinograd" \
    --split test \
    --replace_method percent \
    --operation_non_target ".1" \
    --operation_target "=10" \
    --metrics "acc" \
    --kaggle_dataname_to_save "accxx-xwinograd-ap-d" \
    --parent_dir_to_save "" \
    --target_langs 0 7 5 8 9 6 \
    --is_update \
    > sea_xwinogradxx.txt 2>&1