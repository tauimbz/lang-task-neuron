python get_activations.py \
    --hf_logintoken "***REMOVED***" \
    --model "google/gemma-2-2b-it" \
    --dataset_name "Muennighoff/flores200" \
    --split "dev" \
    --max_instances 1000 \
    --max_tokens_overzeros 100000 \
    --kaggle_dataname_to_save "activationxx-gemma2-neurons" \
    --selected_langs "als_Latn" "arb_Arab" "hye_Armn" "azj_Latn" "eus_Latn" "bel_Cyrl" "ben_Beng" "bul_Cyrl" "zho_Hans" "hrv_Latn" "nld_Latn" "est_Latn" "fin_Latn" "fra_Latn" "kat_Geor" "deu_Latn" "ell_Grek" "heb_Hebr" "hin_Deva" "hun_Latn" "ind_Latn" "ita_Latn" "jpn_Jpan" "kaz_Cyrl" "kor_Hang" "lit_Latn" "zsm_Latn" "mal_Mlym" "npi_Deva" "mkd_Cyrl" "pes_Arab" "pol_Latn" "por_Latn" "rus_Cyrl" "srp_Cyrl" "spa_Latn" "tgl_Latn" "tam_Taml" "tel_Telu" "tur_Latn" "ukr_Cyrl" "urd_Arab" "uzn_Latn" "vie_Latn" \
    --parent_dir_to_save ""\
    > gemma_mini_act3.txt 2>&1
