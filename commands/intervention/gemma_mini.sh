python intervention_dod.py \
    --dataset_kaggle "inayarahmanisa/lsnxx-gemma2-flores" \
    --lsn_filename "gemma2_flores" \
    --ld_filename "lang_dict" \
    --dataset_kaggle_replacer "inayarahmanisa/activationxx-gemma2-flores" \
    --replacer_filename "max.pt" \
    --hf_token "***REMOVED***" \
    --model_name "google/gemma-2-2b-it" \
    --dataset_name "inayarhmns/MLAMA-dod" \
    --langs "en" "nl" "id" "ms" "vi" "jp" "zh" \
    --split "test" \
    --replace_method "percent" \
    --operation_non_target ".1" \
    --operation_target "=10" \
    --metrics "dod" \
    --selected_langs "en" "nl" "id" "ms" "vi" "jp" "zh" \
    --kaggle_dataname_to_save "dod-maplape" \
    --is_update \
    --parent_dir_to_save "" \
    --target_langs 0 1 2 3 4 5 6 
# --show_df_per_lang false \
# --range_layers None \
# --target_langs None \
# --max_samples 2 \
# --apply_template false \

