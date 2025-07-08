python intervention_dod.py \
    --dataset_kaggle "inayarahmanisa/lsnxx-sea1-5-flores" \
    --lsn_filename "raw_act_lsn_sea1.pt" \
    --ld_filename "lang_dict" \
    --dataset_kaggle_replacer "inayarahmanisa/activationxx-sea1-5-neurons" \
    --replacer_filename "median.pt" \
    --hf_token "***REMOVED***" \
    --model_name "SeaLLMs/SeaLLMs-v3-1.5B-Chat" \
    --dataset_name "inayarhmns/MLAMA-dod-185" \
    --split "test" \
    --replace_method "percent" \
    --operation_non_target ".1" \
    --operation_target "=10" \
    --metrics "dod" \
    --kaggle_dataname_to_save "dod-ap" \
    --is_update \
    --langs "en" "nl" "id" "ms" "vi" "ja" "zh" "fr" "pt" "ru" "et" "it" "ta" "th" "tr" \
    --target_langs 0 1 2 3 4 5 6 7 8 9 10 12 15 16 17   \
    --parent_dir_to_save ""
# --show_df_per_lang false \
# --range_layers None \

# --max_samples 2 \
# --apply_template false \

