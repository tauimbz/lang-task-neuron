python intervention_tasks.py \
    --dataset_kaggle "inayarahmanisa/lsnxx-qwen05-flores" \
    --lsn_filename "raw_act_lsn_qwen05.pt" \
    --ld_filename "lang_dict" \
    --dataset_kaggle_replacer "inayarahmanisa/activationxx-qwen05-neurons" \
    --replacer_filename "max.pt" \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --dataset_name "Muennighoff/xwinograd" \
    --split test \
    --replace_method fixed \
    --operation_non_target ".1" \
    --operation_target "=22" \
    --metrics "acc" \
    --kaggle_dataname_to_save "accxx-xwinograd-raw" \
    --parent_dir_to_save "" \
    --target_langs 0 7 5 8 9 6  \
    --is_update \
    --batch_size 8 \
    > qwen_mini_xwinogradxx.txt 2>&1