import torch

# x = torch.load("res/lape/Qwen05B_flores")
# print(x)

y = """"eng_Latn" "nld_Latn" "ind_Latn" "zsm_Latn" "vie_Latn" "jpn_Jpan" "zho_Hans" "fra_Latn" "por_Latn" "rus_Cyrl" "est_Latn" "hat_Latn" "ita_Latn" "quy_Latn" "swh_Latn" "tam_Taml" "tha_Thai" "tur_Latn" """.split(" ")
p = [y[10], y[11], y[2], y[12:], y[4], y[6]]
print(p)

