python get_activations.py  \
    --hf_logintoken "***REMOVED***" \
    --model "SeaLLMs/SeaLLMs-v3-7B-Chat"\
    --dataset_name "Muennighoff/flores200" \
    --split "dev" \
    --max_tokens_overzeros 100000 \
    --max_sentence_avgs 500\
    --selected_langs "eng_Latn" "nld_Latn" "ind_Latn" "zsm_Latn" "vie_Latn" "jpn_Jpan" "zho_Hans"\
    --batch_size 32 \
    --kaggle_dataname_to_save "act-sea7-flores" \
    --parent_dir_to_save "/workspace/" \

    # --max_instances 200 \
    # --is_update \
    # --debug \
    # --take_whole \
    # --max_lang \
    # --selected_langs "deu_Latn" "eng_Latn" "fra_Latn" "ind_Latn" "jpn_Jpan" "kor_Hang" "zsm_Latn" "nld_Latn" "por_Latn" "rus_Cyrl" "vie_Latn" "zho_Hans"\
    # --is_predict \