
python get_map_neurons.py  \
    --in_kaggle \
    --dataset_kaggle "inayarahmanisa/activationxx-sea1-5-neurons"\
    --filename "act_flores200_997_False.pt" \
    --num_layer 28 \
    --model_name_inf "sea1-5" \
    --dataset_name_inf "flores" \
    --kaggle_dataname_to_save "lsnxx-sea1-5-flores" \
    --threshold 0.99 0.95 0.90 0.3 \
    --top_k 1085 \
    --is_update