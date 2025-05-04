python get_activations.py  \
    --hf_logintoken "***REMOVED***" \
    --model "Qwen/Qwen2.5-0.5B-Instruct"\
    --dataset_name "Muennighoff/flores200" \
    --split "devtest" \
    --max_instances 2 \
    --apply_template \
    --max_tokens_overzeros 100000\
    --max_sentence_avgs 500\
    --kaggle_dataname_to_save "act-qwen05-neurons" \
    --selected_langs "deu_Latn" "eng_Latn" "fra_Latn" "ind_Latn" "jpn_Jpan" "kor_Hang" "zsm_Latn" "nld_Latn" "por_Latn" "rus_Cyrl" "vie_Latn" "zho_Hans"\
    --is_update \
    # --debug \
    # --take_whole \
    # --max_lang \
    # --selected_langs "deu_Latn" "eng_Latn" "fra_Latn" "ind_Latn" "jpn_Jpan" "kor_Hang" "zsm_Latn" "nld_Latn" "por_Latn" "rus_Cyrl" "vie_Latn" "zho_Hans"\
    # --is_predict \