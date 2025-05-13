python3 get_activations.py  \
    --hf_logintoken "***REMOVED***" \
    --model "google/gemma-2-9b-it" \
    --dataset_name "Muennighoff/flores200" \
    --split "dev" \
    --max_tokens_overzeros 100000 \
    --max_sentence_avgs 100 \
    --selected_langs "eng_Latn" "nld_Latn" "ind_Latn" "zsm_Latn" "vie_Latn" "jpn_Jpan" "zho_Hans" "fra_Latn" "por_Latn" "rus_Cyrl" "est_Latn" "hat_Latn" "ita_Latn" "quy_Latn" "swh_Latn" "tam_Taml" "tha_Thai" "tur_Latn" \
    --batch_size 16 \
    --kaggle_dataname_to_save "activationxx-gemma9-flores" \
    --parent_dir_to_save "workspace/" \
    > gemma9_actxx.txt 2>&1
    

# python3 get_activations.py  \
#     --hf_logintoken "***REMOVED***" \
#     --model "google/gemma-2-9b-it"\
#     --dataset_name "Muennighoff/flores200" \
#     --split "dev" \
#     --max_tokens_overzeros 100000 \
#     --max_sentence_avgs 500\
#     --selected_langs "eng_Latn" "nld_Latn" "ind_Latn" "zsm_Latn" "vie_Latn" "jpn_Jpan" "zho_Hans"\
#     --batch_size 16 \
#     --kaggle_dataname_to_save "activation-gemma9-flores" \
#     --parent_dir_to_save "/workspace/" \
#     > /workspace/log_gemma9.txt 2>&1
    