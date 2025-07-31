#!/bin/bash

python intervention_dod.py \
    --dataset_kaggle "inayarahmanisa/lsn2-gemma2-flores" \
    --lsn_filename "maplape.pt" \
    --ld_filename "lang_dict" \
    --dataset_kaggle_replacer "inayarahmanisa/activation2-gemma2-flores" \
    --replacer_filename "max.pt" \
    --model_name "google/gemma-2-2b-it" \
    --dataset_name "Muennighoff/xwinograd" \
    --split test \
    --replace_method fixed \
    --operation_non_target ".1" \
    --operation_target "=10" \
    --metrics "acc" \
    --kaggle_dataname_to_save "acc2-xwinograd-maplape" \
    --parent_dir_to_save "" \
    --target_langs 0 7 5 8 9 6 \
    --is_update 


python intervention_dod.py \
    --dataset_kaggle "inayarahmanisa/lsn2-gemma2-flores" \
    --lsn_filename "maplape.pt" \
    --ld_filename "lang_dict" \
    --model_name "google/gemma-2-2b-it" \
    --dataset_name "Muennighoff/xwinograd" \
    --split test \
    --replace_method percent \
    --operation_non_target ".1" \
    --operation_target "=10" \
    --metrics "acc" \
    --kaggle_dataname_to_save "acc2-xwinograd-maplape" \
    --parent_dir_to_save "" \
    --target_langs 0 7 5 8 9 6 \
    --is_update 
