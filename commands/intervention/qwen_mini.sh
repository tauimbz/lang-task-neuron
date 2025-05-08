python intervention_dod.py \
    --dataset_kaggle "inayarahmanisa/lsn-qwen05-flores" \
    --lsn_filename "maplape.pt" \
    --ld_filename "lang_dict" \
    --dataset_kaggle_replacer "inayarahmanisa/activation-qwen05-flores" \
    --replacer_filename "max.pt" \
    --hf_token "***REMOVED***" \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
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
    --parent_dir_to_save ""
# --show_df_per_lang false \
# --range_layers None \
# --target_langs None \
# --max_samples 2 \
# --apply_template false \

