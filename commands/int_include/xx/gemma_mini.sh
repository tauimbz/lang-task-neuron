python intervention_tasks.py \
    --dataset_kaggle "inayarahmanisa/lsnxx-gemma2-flores" \
    --lsn_filename "maplape.pt" \
    --ld_filename "lang_dict" \
    --dataset_kaggle_replacer "inayarahmanisa/activationxx-gemma2-neurons" \
    --replacer_filename "max.pt" \
    --hf_token "***REMOVED***" \
    --model_name "google/gemma-2-2b-it" \
    --dataset_name "CohereLabs/include-lite-44" \
    --split test \
    --replace_method fixed \
    --operation_non_target ".1" \
    --operation_target "=10" \
    --metrics "acc" \
    --kaggle_dataname_to_save "accxx-xwinograd-maplape" \
    --parent_dir_to_save "" \
    --target_langs 0 7 5 8 9 6 \
    --is_update \
    --batch_size 8 \
    > gemma_mini_xwinogradxx.txt 2>&1


 "eng_Latn" "nld_Latn" "ind_Latn" "zsm_Latn" "vie_Latn" "jpn_Jpan" "zho_Hans" "fra_Latn" "por_Latn" "rus_Cyrl" "est_Latn" "hat_Latn" "ita_Latn" "quy_Latn" "swh_Latn" "tam_Taml" "tha_Thai" "tur_Latn" 
 1 2 3 4 5 6 7 8 9 10 12 13 14 15 16
 "Dutch" "Indonesian" "Malay" "Vietnamese" "Japanese" "Chinese" "French" "Portuguese" "Russian" "Estonian" "Italian"  "Tamil" "Turkish"

 "English"
"Haitian Creole"
"Quechua"
"Swahili"
"Thai"