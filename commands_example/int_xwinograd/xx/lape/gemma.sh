python3 intervention_tasks.py \
    --dataset_kaggle "inayarahmanisa/lsnxx-gemma9-flores" \
    --lsn_filename "gemma9_flores" \
    --ld_filename "lang_dict" \
    --dataset_kaggle_replacer "inayarahmanisa/activationxx-gemma9-flores" \
    --replacer_filename "max.pt" \
    --model_name "google/gemma-2-9b-it" \
    --dataset_name "Muennighoff/xwinograd" \
    --split test \
    --replace_method fixed \
    --operation_non_target ".1" \
    --operation_target "=10" \
    --metrics "acc" \
    --kaggle_dataname_to_save "accxx-xwinograd-lape" \
    --parent_dir_to_save "" \
    --target_langs 0 7 5 8 9 6 \
    --is_update \
    --batch_size 4 \
    > gemma_xwinogradxx.txt 2>&1
