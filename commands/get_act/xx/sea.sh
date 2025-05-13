python3 get_activations.py  \
    --hf_logintoken "***REMOVED***" \
    --model "SeaLLMs/SeaLLMs-v3-7B-Chat"\
    --dataset_name "Muennighoff/flores200" \
    --split "dev" \
    --max_tokens_overzeros 100000 \
    --max_sentence_avgs 100\
    --selected_langs "eng_Latn" "nld_Latn" "ind_Latn" "zsm_Latn" "vie_Latn" "jpn_Jpan" "zho_Hans" "fra_Latn" "por_Latn" "rus_Cyrl" "est_Latn" "hat_Latn" "ita_Latn" "quy_Latn" "swh_Latn" "tam_Taml" "tha_Thai" "tur_Latn" \
    --batch_size 64 \
    --kaggle_dataname_to_save "activationxx-sea7-flores" \
    --parent_dir_to_save "/workspace/" \
    > sea7_actxx.txt 2>&1
