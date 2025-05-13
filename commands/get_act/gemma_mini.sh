nohup python get_activations.py \
    --hf_logintoken "***REMOVED***" \
    --model "google/gemma-2-2b-it" \
    --dataset_name "Muennighoff/flores200" \
    --split "dev" \
    --max_instances 1000 \
    --apply_template \
    --max_tokens_overzeros 100000 \
    --kaggle_dataname_to_save "act-gemma2-neurons" \
    --selected_langs "deu_Latn" "eng_Latn" "fra_Latn" "ind_Latn" "jpn_Jpan" "kor_Hang" "zsm_Latn" "nld_Latn" "por_Latn" "rus_Cyrl" "vie_Latn" "zho_Hans" \
    --parent_dir_to_save "/workspace/"\
    > output.log 2>&1
