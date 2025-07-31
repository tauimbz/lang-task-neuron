python intervention_dod.py \
    --dataset_kaggle "inayarahmanisa/lsnxx-gemma2-flores" \
    --lsn_filename "gemma2_flores" \
    --ld_filename "lang_dict" \
    --dataset_kaggle_replacer "inayarahmanisa/activationxx-gemma2-neurons" \
    --replacer_filename "max.pt" \
    --model_name "google/gemma-2-2b-it" \
    --dataset_name "inayarhmns/MLAMA-dod-185" \
    --split "test" \
    --replace_method "percent" \
    --operation_non_target ".1" \
    --operation_target "=10" \
    --range_layers 9 \
    --metrics "dod" \
    --kaggle_dataname_to_save "perlayer-dod-lape" \
    --is_update \
    --target_langs 0 1 2 3 4 5 6 7 8 9 10 12 15 16 17   \
    --langs "en" "nl" "id" "ms" "vi" "ja" "zh" "fr" "pt" "ru" "et" "it" "ta" "th" "tr" \
    --parent_dir_to_save "" 
    
# --show_df_per_lang false \
# --range_layers None \
# --target_langs None \
# --max_samples 2 \
# --apply_template false \

