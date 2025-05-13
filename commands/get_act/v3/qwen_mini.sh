python get_activations.py  \
    --hf_logintoken "***REMOVED***" \
    --model "Qwen/Qwen2.5-0.5B-Instruct" \
    --dataset_name "Muennighoff/flores200" \
    --split "dev" \
    --apply_template \
    --max_tokens_overzeros 100000 \
    --selected_langs "als_Latn" "arb_Arab" "hye_Armn" "azj_Latn" "eus_Latn" "bel_Cyrl" "ben_Beng" "bul_Cyrl" "zho_Hans" "hrv_Latn" "nld_Latn" "est_Latn" "fin_Latn" "fra_Latn" "kat_Geor" "deu_Latn" "ell_Grek" "heb_Hebr" "hin_Deva" "hun_Latn" "ind_Latn" "ita_Latn" "jpn_Jpan" "kaz_Cyrl" "kor_Hang" "lit_Latn" "zsm_Latn" "mal_Mlym" "npi_Deva" "mkd_Cyrl" "pes_Arab" "pol_Latn" "por_Latn" "rus_Cyrl" "srp_Cyrl" "spa_Latn" "tgl_Latn" "tam_Taml" "tel_Telu" "tur_Latn" "ukr_Cyrl" "urd_Arab" "uzn_Latn" "vie_Latn" \
    --max_instances 1000 \
    --batch_size 64 \
    --kaggle_dataname_to_save "activation3-qwen05-neurons" \
    > qwen_mini_act3.txt 2>&1
# --debug \
# --take_whole \
# --max_lang \
# --selected_langs "deu_Latn" "eng_Latn" "fra_Latn" "ind_Latn" "jpn_Jpan" "kor_Hang" "zsm_Latn" "nld_Latn" "por_Latn" "rus_Cyrl" "vie_Latn" "zho_Hans"\
# --is_predict \
# --parent_dir_to_save "/workspace/"