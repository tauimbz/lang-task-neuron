python intervention_tasks.py \
    --dataset_kaggle "inayarahmanisa/lsnxx-gemma9-flores" \
    --lsn_filename "gemma9_flores" \
    --ld_filename "lang_dict" \
    --dataset_kaggle_replacer "inayarahmanisa/activationxx-gemma9-flores" \
    --replacer_filename "max.pt" \
    --model_name "google/gemma-2-9b-it" \
    --dataset_name "Muennighoff/flores200" \
    --split devtest \
    --replace_method fixed \
    --operation_non_target ".1" \
    --operation_target "=10" \
    --metrics "ppl_full" \
    --kaggle_dataname_to_save "ppl-flores-da" \
    --parent_dir_to_save "" \
    --selected_langs "eng_Latn" "nld_Latn" "ind_Latn" "zsm_Latn" "vie_Latn" "jpn_Jpan" "zho_Hans" "fra_Latn" "por_Latn" "rus_Cyrl" "est_Latn" "hat_Latn" "ita_Latn" "quy_Latn" "swh_Latn" "tam_Taml" "tha_Thai" "tur_Latn" \
    --target_langs 0 1 2 3 4 5 6  7 8 9 10 11 12 13 14 15 16 17 \
    --is_update \
    --batch_size 8 \
    > gemma_ppl_lape.txt 2>&1