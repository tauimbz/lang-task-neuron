python3 get_activations.py  \
    --hf_logintoken "***REMOVED***" \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --dataset_name "Muennighoff/flores200" \
    --split "dev" \
    --max_tokens_overzeros 100000 \
    --max_sentence_avgs 100\
    --selected_langs "eng_Latn" "nld_Latn" "ind_Latn" "zsm_Latn" "vie_Latn" "jpn_Jpan" "zho_Hans" "fra_Latn" "por_Latn" "rus_Cyrl" "est_Latn" "hat_Latn" "ita_Latn" "quy_Latn" "swh_Latn" "tam_Taml" "tha_Thai" "tur_Latn" \
    --batch_size 32 \
    --kaggle_dataname_to_save "activationxx-qwen7-flores" \
    --parent_dir_to_save "/workspace/" \
    > qwen7_actxx.txt 2>&1
# --max_instances 200 \
# --is_update \
# --debug \
# --take_whole \
# --max_lang \
# --selected_langs "deu_Latn" "eng_Latn" "fra_Latn" "ind_Latn" "jpn_Jpan" "kor_Hang" "zsm_Latn" "nld_Latn" "por_Latn" "rus_Cyrl" "vie_Latn" "zho_Hans"\
# --is_predict \