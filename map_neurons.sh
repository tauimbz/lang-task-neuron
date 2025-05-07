python get_map_neurons.py  \
    --in_kaggle \
    --dataset_kaggle "inayarahmanisa/activation-qwen05-flores"\
    --filename "act_flores200_997_False.pt" \
    --num_layer 24 \
    --model_name_inf "Qwen05B" \
    --dataset_name_inf "flores" \
    --kaggle_dataname_to_save "lsn-qwen05-flores" \
    --threshold 0.99 0.95 0.90 0.8 0.7 0.6 0.5 0.3 \
    --top_bottom_k 1000 \
    --is_update \
    --subdir_name "map" 

python get_map_neurons.py  \
    --in_kaggle \
    --dataset_kaggle "inayarahmanisa/activation-qwen7-flores"\
    --filename "act_flores200_997_False.pt" \
    --num_layer 28 \
    --model_name_inf "Qwen7B" \
    --dataset_name_inf "flores" \
    --kaggle_dataname_to_save "lsn-qwen7-flores" \
    --threshold 0.99 0.95 0.90 0.8 0.7 0.6 0.5 0.3 \
    --top_bottom_k 1000 \
    --is_update \
    --subdir_name "map" 
    
python get_map_neurons.py  \
    --in_kaggle \
    --dataset_kaggle "inayarahmanisa/activation-gemma2-flores"\
    --filename "act_flores200_997_False.pt" \
    --num_layer 26 \
    --model_name_inf "gemma2" \
    --dataset_name_inf "flores" \
    --kaggle_dataname_to_save "lsn-gemma2-flores" \
    --threshold 0.99 0.95 0.90 0.8 0.7 0.6 0.5 0.3 \
    --top_bottom_k 1000 \
    --is_update \
    --subdir_name "map" 



python get_map_neurons.py  \
    --in_kaggle \
    --dataset_kaggle "inayarahmanisa/activation-sea1-5-flores"\
    --filename "act_flores200_997_False.pt" \
    --num_layer 28 \
    --model_name_inf "sea1-5" \
    --dataset_name_inf "flores" \
    --kaggle_dataname_to_save "lsn-sea1-5-flores" \
    --threshold 0.99 0.95 0.90 0.8 0.7 0.6 0.5 0.3 \
    --top_bottom_k 1000 \
    --is_update \
    --subdir_name "map" 


python get_map_neurons.py  \
    --in_kaggle \
    --dataset_kaggle "inayarahmanisa/activation-sea7-flores"\
    --filename "act_flores200_997_False.pt" \
    --num_layer 28 \
    --model_name_inf "sea7" \
    --dataset_name_inf "flores" \
    --kaggle_dataname_to_save "lsn-sea7-flores" \
    --threshold 0.99 0.95 0.90 0.8 0.7 0.6 0.5 0.3 \
    --top_bottom_k 1000 \
    --is_update \
    --subdir_name "map" 

python get_map_neurons.py  \
    --in_kaggle \
    --dataset_kaggle "inayarahmanisa/activation-gemma9-flores"\
    --filename "act_flores200_997_False.pt" \
    --num_layer 42 \
    --model_name_inf "gemma9" \
    --dataset_name_inf "flores" \
    --kaggle_dataname_to_save "lsn-gemma9-flores" \
    --threshold 0.99 0.95 0.90 0.8 0.7 0.6 0.5 0.3 \
    --top_bottom_k 1000 \
    --is_update \
    --subdir_name "map" 











# BATAS



# python get_map_neurons.py  \
#     --in_kaggle \
#     --dataset_kaggle "inayarahmanisa/activation-qwen05-flores"\
#     --filename "act_flores200_997_False.pt" \
#     --num_layer 24 \
#     --model_name_inf "Qwen05B" \
#     --dataset_name_inf "flores" \
#     --kaggle_dataname_to_save "lsn-qwen05-flores" \
#     --threshold 0.99 0.95 0.90 0.8 0.7 0.6 0.5 0.3 \
#     --top_bottom_k 1000 \
#     --is_update \
#     --subdir_name "map" \
#     # --result_exist \
#     # --data_kaggle_result "nayechz/neuron-xwinograd"\
#     # --res_filename "raw_qwen05_xwinograd_83.pt" \
#     # --parent_dir_to_save "/workspace/"\
#     # --is_update \
#     # --is_last_token False \
#     # --max_instances None \
#     # --kaggle_dataname_to_save None \
#     # --is_update None
    


# python get_map_neurons.py  \
#     --in_kaggle \
#     --dataset_kaggle "nayechz/neuron-xwinograd" \
#     --filename "raw_qwen05_xwinograd_83.pt" \
#     --num_layer 24 \
#     --model_name_inf "Qwen05B" \
#     --dataset_name_inf "xwinograd" \
#     --kaggle_dataname_to_save "tesdrive-xwin-map-qwen05" \
#     --top_k 1000 \
#     --result_exist \
#     --data_kaggle_result "inayarahmanisa/tesdrive-xwin-map-qwen05"\
#     --res_filename "result_Qwen05B_xwinograd.pt" \
#     --is_update 
#     # --parent_dir_to_save "/workspace/"\
#     # --is_update \
#     # --is_last_token False \
#     # --max_instances None \
#     # --kaggle_dataname_to_save None \
#     # --is_update None


# python get_map_neurons.py  \
#     --in_kaggle \
#     --dataset_kaggle "inayarahmanisa/activation-qwen05-flores"\
#     --filename "act_flores200_997_False.pt" \
#     --num_layer 24 \
#     --model_name_inf "Qwen05B" \
#     --dataset_name_inf "flores" \
#     --kaggle_dataname_to_save "lsn-qwen05-flores" \
#     --threshold 0.99 0.5 0.3 \
#     --top_bottom_k 1000 \
#     --is_update \
#     # --result_exist \
#     # --data_kaggle_result "nayechz/neuron-xwinograd"\
#     # --res_filename "raw_qwen05_xwinograd_83.pt" \
#     # --parent_dir_to_save "/workspace/"\
#     # --is_update \
#     # --is_last_token False \
#     # --max_instances None \
#     # --kaggle_dataname_to_save None \
#     # --is_update None
    