python get_activations.py  \
    --hf_logintoken "***REMOVED***" \
    --model "Qwen/Qwen2.5-0.5B-Instruct" \
    --dataset_name "Muennighoff/flores200" \
    --split "dev" \
    --apply_template \
    --max_tokens_overzeros 100000 \
    --selected_langs "eng_Latn" "nld_Latn" "ind_Latn" "zsm_Latn" "vie_Latn" "jpn_Jpan" "zho_Hans" "fra_Latn" "por_Latn" "rus_Cyrl" "est_Latn" "hat_Latn" "ita_Latn" "quy_Latn" "swh_Latn" "tam_Taml" "tha_Thai" "tur_Latn" \
    --max_instances 1000 \
    --batch_size 32 \
    --kaggle_dataname_to_save "activationxx-qwen05-neurons" \
    > qwen_mini_actxx.txt 2>&1
# --debug \
# --take_whole \
# --max_lang \
# --selected_langs "deu_Latn" "eng_Latn" "fra_Latn" "ind_Latn" "jpn_Jpan" "kor_Hang" "zsm_Latn" "nld_Latn" "por_Latn" "rus_Cyrl" "vie_Latn" "zho_Hans"\
# --is_predict \
# --parent_dir_to_save "/workspace/"