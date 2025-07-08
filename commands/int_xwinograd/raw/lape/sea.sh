python intervention_tasks.py \
    --dataset_kaggle "inayarahmanisa/lsnxx-sea7-flores" \
    --lsn_filename "sea7_flores" \
    --ld_filename "lang_dict" \
    --dataset_kaggle_replacer "inayarahmanisa/activationxx-sea7-flores" \
    --replacer_filename "max.pt" \
    --hf_token "***REMOVED***" \
    --model_name "SeaLLMs/SeaLLMs-v3-7B-Chat" \
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
    --batch_size 32 \
    > sea_xwinogradxx.txt 2>&1