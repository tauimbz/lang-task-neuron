python intervention_dod.py \
    --dataset_kaggle "inayarahmanisa/lsnxx-gemma2-flores" \
    --lsn_filename "maplape.pt" \
    --ld_filename "lang_dict" \
    --dataset_kaggle_replacer "inayarahmanisa/activationxx-gemma2-flores" \
    --replacer_filename "max.pt" \
    --hf_token "***REMOVED***" \
    --model_name "google/gemma-2-2b-it" \
    --dataset_name "Muennighoff/xwinograd" \
    --split test \
    --replace_method fixed \
    --operation_non_target ".1" \
    --operation_target "=10" \
    --metrics "acc" \
    --kaggle_dataname_to_save "accxx-xwinograd-maplape" \
    --parent_dir_to_save "" \
    --target_langs 0 7 5 8 9 6 \
    --is_update \
    > gemma_mini_xwinogradxx.txt 2>&1
