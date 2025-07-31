python intervention_tasks.py \
    --dataset_kaggle "inayarahmanisa/lsnxx-sea1-5-flores" \
    --lsn_filename "raw_act_lsn_sea1.pt" \
    --ld_filename "lang_dict" \
    --dataset_kaggle_replacer "inayarahmanisa/activationxx-sea1-5-neurons" \
    --replacer_filename "max.pt" \
    --model_name "SeaLLMs/SeaLLMs-v3-1.5B-Chat" \
    --dataset_name "facebook/flores" \
    --split devtest \
    --replace_method fixed \
    --operation_non_target ".1" \
    --operation_target "=10" \
    --metrics "bleu" \
    --kaggle_dataname_to_save "bleu-flores-raw" \
    --parent_dir_to_save "" \
    --selected_langs "eng_Latn-nld_Latn" "eng_Latn-ind_Latn" "eng_Latn-zsm_Latn" "eng_Latn-vie_Latn" "eng_Latn-jpn_Jpan" "eng_Latn-zho_Hans" "eng_Latn-fra_Latn" "eng_Latn-por_Latn" "eng_Latn-rus_Cyrl" "eng_Latn-est_Latn" "eng_Latn-hat_Latn" "eng_Latn-ita_Latn" "eng_Latn-quy_Latn" "eng_Latn-swh_Latn" "eng_Latn-tam_Taml" "eng_Latn-tha_Thai" "eng_Latn-tur_Latn" \
    --target_langs 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 \
    --is_update \
    --batch_size 8 \
    > sea_bleu_raw.txt 2>&1