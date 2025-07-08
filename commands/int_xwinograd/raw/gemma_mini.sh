python intervention_tasks.py \
    --dataset_kaggle "inayarahmanisa/lsnxx-gemma2-neurons" \
    --lsn_filename "raw_act_lsn_gemma2.pt" \
    --ld_filename "ld_flores200" \
    --dataset_kaggle_replacer "inayarahmanisa/activationxx-gemma2-neurons" \
    --replacer_filename "max.pt" \
    --hf_token "***REMOVED***" \
    --model_name "google/gemma-2-2b-it" \
    --dataset_name "Muennighoff/xwinograd" \
    --split test \
    --replace_method fixed \
    --operation_non_target ".1" \
    --operation_target "=22" \
    --metrics "acc" \
    --kaggle_dataname_to_save "accxx-xwinograd-raw" \
    --parent_dir_to_save "" \
    --target_langs 0 7 5 8 9 6 \
    --is_update \
    --batch_size 4 \
    > gemma_mini_xwinogradxx.txt 2>&1
