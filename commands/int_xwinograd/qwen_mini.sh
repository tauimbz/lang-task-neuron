python intervention_dod.py \
    --dataset_kaggle "inayarahmanisa/lsn2-qwen05-flores" \
    --lsn_filename "maplape.pt" \
    --ld_filename "lang_dict" \
    --dataset_kaggle_replacer "inayarahmanisa/activation2-qwen05-flores" \
    --replacer_filename "max.pt" \
    --hf_token "***REMOVED***" \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
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
    --dataset_kaggle "inayarahmanisa/lsn2-qwen05-flores" \
    --lsn_filename "maplape.pt" \
    --ld_filename "lang_dict" \
    --hf_token "***REMOVED***" \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
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
