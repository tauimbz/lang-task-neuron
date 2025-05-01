python get_activations.py  \
    --hf_logintoken "***REMOVED***" \
    --model "Qwen/Qwen2.5-0.5B-Instruct"\
    --dataset_name "Muennighoff/flores200" \
    --split "devtest" \
    --max_instances 1 \
    --apply_template \
    --debug \
    --max_tokens_overzeros 10000\
    # --take_whole \
    # --kaggle_dataname_to_save None \
    # --is_update None
    # --max_lang \
    # --selected_langs \
    # --is_predict \