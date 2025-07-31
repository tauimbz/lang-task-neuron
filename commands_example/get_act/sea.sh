python3 get_activations.py  \
    --model "SeaLLMs/SeaLLMs-v3-7B-Chat"\
    --dataset_name "Muennighoff/flores200" \
    --split "dev" \
    --max_tokens_overzeros 100000 \
    --max_sentence_avgs 100\
    --selected_langs "eng_Latn" "nld_Latn" "ind_Latn" "zsm_Latn" "vie_Latn" "jpn_Jpan" "zho_Hans"\
    --batch_size 16 \
    --kaggle_dataname_to_save "activation-sea7-flores" \
    --parent_dir_to_save "workspace/" \
    > workspace/log_sea7.txt 2>&1
