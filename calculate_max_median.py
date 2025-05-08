import torch
from kaggle_utils import *
import argparse

parser = argparse.ArgumentParser(description="Run the map function with specified parameters.")
parser.add_argument("--in_kaggle", action='store_true', help="Whether the file is from Kaggle")
parser.add_argument("--dataset_kaggle", type=str,  help="The dataset name in Kaggle, if not Kaggle then empty")
parser.add_argument("--filename", type=str, help="The filename to process")
parser.add_argument("--kaggle_dataname_to_save", type=str, default=None, help="Dataset name for saving to Kaggle NO USERNAME!")
parser.add_argument("--is_update", action='store_true', help="Flag to update Kaggle dataset")
parser.add_argument("--n_layer", type=int, help="num layer")
parser.add_argument("--parent_dir_to_save", type=str, default=None, help="Parent directory to save like /workspace for runpod")

args = parser.parse_args()

#MODIF

# args, unknown = parser.parse_known_args()
# args.in_kaggle = True
# args.dataset_kaggle = "inayarahmanisa/activation-qwen05-flores"
# args.filename = "act_flores200_997_False.pt"
# args.kaggle_dataname_to_save = "activation-qwen05-flores"
# args.is_update = True
# args.n_layer = 24
#END MODIF



parent_dir_to_save = args.parent_dir_to_save if args.parent_dir_to_save else ""


download_from_kaggle(args.dataset_kaggle, args.filename)
act_file = torch.load(f"data/{args.filename}", weights_only=True)

max_vals, max_indices = act_file.max(dim=1)  
max_vals = max_vals.reshape(act_file.shape[0], args.n_layer, int(act_file.shape[-1]/args.n_layer))

median_vals, median_indices = act_file.median(dim=1)
median_vals = median_vals.reshape(act_file.shape[0], args.n_layer, int(act_file.shape[-1]/args.n_layer))

path_res = f"{parent_dir_to_save}res"
os.makedirs(path_res, exist_ok=True)


torch.save(median_vals, f"{path_res}/median.pt")
torch.save(max_vals, f"{path_res}/max.pt")


save_to_kaggle(dataset_name=args.kaggle_dataname_to_save, data_dir=path_res, is_update=args.is_update)