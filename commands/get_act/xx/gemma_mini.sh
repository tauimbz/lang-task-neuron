python get_activations.py \
    --hf_logintoken "***REMOVED***" \
    --model "google/gemma-2-2b-it" \
    --dataset_name "Muennighoff/flores200" \
    --split "dev" \
    --max_sentence_avgs 100 \
    --max_tokens_overzeros 100000 \
    --kaggle_dataname_to_save "activation3-gemma2-neurons" \
    --selected_langs "eng_Latn" "nld_Latn" "ind_Latn" "zsm_Latn" "vie_Latn" "jpn_Jpan" "zho_Hans" "fra_Latn" "por_Latn" "rus_Cyrl" "est_Latn" "hat_Latn" "ita_Latn" "quy_Latn" "swh_Latn" "tam_Taml" "tha_Thai" "tur_Latn" \
    --parent_dir_to_save ""\
    --batch_size 32 \
    > gemma_mini_actxx.txt 2>&1

