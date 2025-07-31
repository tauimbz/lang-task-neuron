python intervention_tasks.py \
    --dataset_kaggle "inayarahmanisa/lsnxx-qwen7-xwinograd" \
    --lsn_filename "qwen7_flores" \
    --ld_filename "lang_dict" \
    --dataset_kaggle_replacer "inayarahmanisa/activation-xwin-qwen7-neurons" \
    --replacer_filename "max.pt" \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --dataset_name "Muennighoff/xwinograd" \
    --split test \
    --replace_method fixed \
    --operation_non_target ".1" \
    --operation_target "=22" \
    --metrics "acc" \
    --kaggle_dataname_to_save "accxx-xwinograd-xwinograd" \
    --parent_dir_to_save "" \
    --is_update \
    --batch_size 8 \
    > qwen_xwinogradxx.txt 2>&1