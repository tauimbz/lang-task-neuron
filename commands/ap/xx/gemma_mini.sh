python get_map_neurons.py  \
    --dataset_kaggle "inayarahmanisa/activationxx-gemma2-flores"\
    --filename "act_flores200_997_False.pt" \
    --num_layer 26 \
    --model_name_inf "gemma2" \
    --dataset_name_inf "flores" \
    --kaggle_dataname_to_save "lsnxx-gemma2-flores" \
    --threshold 0.99 0.95 0.90 0.3 \
    --top_bottom_k 1000 \
    --is_update
