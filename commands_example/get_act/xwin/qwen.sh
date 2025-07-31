python3 get_activations.py  \
    --model "Qwen/Qwen2.5-7B-Instruct"\
    --dataset_name "Muennighoff/xwinograd" \
    --split "test" \
    --max_tokens_overzeros 100000 \
    --max_sentence_avgs 83\
    --batch_size 16 \
    --kaggle_dataname_to_save "activation-xwin-qwen7-neurons" \
    --parent_dir_to_save "" \
    --is_update \
    > log_qwen7.txt 2>&1
    # --max_instances 200 \
    # --is_update \
    # --debug \
    # --take_whole \
    # --max_lang \
    # --selected_langs "deu_Latn" "eng_Latn" "fra_Latn" "ind_Latn" "jpn_Jpan" "kor_Hang" "zsm_Latn" "nld_Latn" "por_Latn" "rus_Cyrl" "vie_Latn" "zho_Hans"\
    # --is_predict \