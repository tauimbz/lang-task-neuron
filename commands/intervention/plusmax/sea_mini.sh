python intervention_dod.py \
    --dataset_kaggle "inayarahmanisa/lsnxx-sea1-5-flores" \
    --lsn_filename "map_t1085_sea1-5_flores.pt" \
    --ld_filename "lang_dict" \
    --hf_token "***REMOVED***" \
    --model_name "SeaLLMs/SeaLLMs-v3-1.5B-Chat" \
    --dataset_name "inayarhmns/MLAMA-dod-185" \
    --split "test" \
    --replace_method "max" \
    --operation_non_target ".1" \
    --operation_target "+1" \
    --metrics "dod" \
    --kaggle_dataname_to_save "dod-ap" \
    --is_update \
    --target_langs 0 1 2 3 4 5 6 7 8 9 10 12 15 16 17   \
    --langs "en" "nl" "id" "ms" "vi" "ja" "zh" "fr" "pt" "ru" "et" "it" "ta" "th" "tr" \
    --parent_dir_to_save ""
# --show_df_per_lang false \
# --range_layers None \
# --target_langs None \
# --max_samples 2 \
# --apply_template false \

