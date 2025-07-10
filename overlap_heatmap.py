# code overlap lape

from raw_act_neurons import *
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np




import argparse
from kaggle_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_kaggle_result", type=str, default=None, help="dataset name if result_exist nd is in Kaggle dataset")
parser.add_argument("--lsn_filename", type=str, default=None, help="filename that saves lsn")
parser.add_argument("--ld_filename", type=str, default=None, help="filename that saves ld")
parser.add_argument("--save", action='store_true', help="Flag to save heatmap")
parser.add_argument("--alpha", type=int, default=None, help="alpha for jaccard")
parser.add_argument("--annot_kws", type=int, default=None, help="font size")
parser.add_argument("--n_layer", type=int, default=None, help="number of layer")
parser.add_argument("--n_neurons", type=int, default=None, help="number of neuron per layer")
parser.add_argument("--parent_dir_to_save", type=str, default=None, help="Parent directory to save like /workspace for runpod")
parser.add_argument("--kaggle_dataname_to_save", type=str, default=None, help="Dataset name for saving to Kaggle NO USERNAME!")


args = parser.parse_args()

data_kaggle_result = args.data_kaggle_result
model_name_inf = data_kaggle_result.split("-")[1]
lsn_filename = args.lsn_filename
annot_kws = args.annot_kws
download_from_kaggle(data_kaggle_result, lsn_filename)
lsn = torch.load(f"data/{lsn_filename}")
# num_lang, num_sentences, total_neuron = lsn.shape
# num_lang = len(lsn)
# num_layer = args.n_layer
# num_neuron = total_neuron/num_layer


ld_filename = args.ld_filename
download_from_kaggle(data_kaggle_result, ld_filename)
ld = torch.load(f"data/{ld_filename}")

alpha = args.alpha if args.alpha else 1



activation_dict = dict()
# data: list of 18 elements (languages)
# each language: list of 24 tensors (one per layer) of neuron indices from 0–4863

data=lsn
num_langs = len(data)
num_layers = args.n_layer
neurons_per_layer = args.n_neurons

# Step 1: Adjust neuron indices so each layer's neurons are globally unique
adjusted_lang_sets = []
idx_lang = 0
for lang in data:
    full_set = set()
    for layer_idx, layer_tensor in enumerate(lang):
        offset = layer_idx * neurons_per_layer
        adjusted = (layer_tensor + offset).tolist()
        full_set.update(adjusted)
    adjusted_lang_sets.append(full_set)
    activation_dict[idx_lang] = list(full_set)
    idx_lang += 1

# make_heatmap_neuron_overlap(activation_dict, k=18, with_label=True, alpha=1, method="default", lang_dict=ld, save=True, save_name="overlap_lape")


make_heatmap_neuron_overlap(activation_dict, num_langs, False)
make_heatmap_neuron_overlap(activation_dict, k=num_langs, with_label=True, alpha=alpha, method="default", lang_dict=ld, save=args.save, modelname=model_name_inf, annot_kws=annot_kws)
make_heatmap_neuron_overlap(activation_dict, k=num_langs, with_label=True, alpha=alpha, method="jaccard", lang_dict=ld, save=args.save, modelname=model_name_inf, annot_kws=annot_kws+4)
