python get_activations.py  \
    --hf_logintoken "***REMOVED***" \
    --model "Qwen/Qwen2.5-7B-Instruct"\
    --dataset_name "Muennighoff/flores200" \
    --split "dev" \
    --apply_template \
    --max_tokens_overzeros 100000\
    --kaggle_dataname_to_save "act-qwen7-neurons" \
    --selected_langs "deu_Latn" "eng_Latn" "fra_Latn" "ind_Latn" "jpn_Jpan" "kor_Hang" "zsm_Latn" "nld_Latn" "por_Latn" "rus_Cyrl" "vie_Latn" "zho_Hans"\
    --batch_size 32 \
    # --parent_dir_to_save "/workspace" \
    # --max_instances 1000 \
    # --is_update \
    # --debug \
    # --take_whole \
    # --max_lang \
    # --selected_langs "deu_Latn" "eng_Latn" "fra_Latn" "ind_Latn" "jpn_Jpan" "kor_Hang" "zsm_Latn" "nld_Latn" "por_Latn" "rus_Cyrl" "vie_Latn" "zho_Hans"\
    # --is_predict \

