from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from model import InferenceModel
from datasets import load_dataset, get_dataset_config_names
import pandas as pd
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from tqdm import tqdm
import argparse
from kaggle_utils import *
from data_sets import *


parser = argparse.ArgumentParser()
parser.add_argument("--hf_logintoken", type=str, default=None, help="HF Login")
parser.add_argument("-m", "--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")

parser.add_argument("--dataset_name", type=str, default="AmazonScience/massive", help="Dataset to use")
parser.add_argument("--split", type=str, default="test", help="Dataset split")
parser.add_argument("--max_instances", type=int, default=None, help="Maximum number of instances")
parser.add_argument("--max_lang", type=int, default=None, help="Maximum number of languages to use")
parser.add_argument("--selected_langs", nargs='+', default=None, help="List of selected languages")
parser.add_argument("--is_predict", action="store_true", help="Whether to run predictions")
parser.add_argument("--apply_template", action="store_true", help="Whether to apply template")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--take_whole", action="store_true", help="Take whole neuron from prompt if is_predict True")
parser.add_argument("--max_tokens_overzeros", type=int, default=10000, help="Max tokens overzeros")
parser.add_argument("--kaggle_dataname_to_save", type=str, default=None, help="Dataset name for saving to Kaggle NO USERNAME!")
parser.add_argument("--is_update", action='store_true', help="Flag to update Kaggle dataset")
parser.add_argument("--parent_dir_to_save", type=str, default=None, help="Parent directory to save like /workspace for runpod")

args = parser.parse_args()



# MODIF FOR DEBUG
# args = parser.parse_args()
# args, unknown = parser.parse_known_args()
# args.model = "Qwen/Qwen2.5-0.5B-Instruct"
# args.hf_logintoken = "***REMOVED***"
# args.dataset_name = "Muennighoff/flores200"
# args.split = "devtest"
# args.apply_template = True
# args.max_tokens_overzeros = 40
# args.selected_langs = [
#     "deu_Latn", "eng_Latn"]
# args.max_instances = 2
# args.batch_size = 32
# END MODIF


parent_dir = args.parent_dir_to_save if args.parent_dir_to_save else ""
login(args.hf_logintoken)
model_name = args.model

infer_model = InferenceModel(model_name)
raw_values_avg_tokens = {}
# raw_values_last_token = {}
full_raw_values_avg_tokens = []
# full_raw_values_last_token = []
# full_raw_values = []
num_layers = infer_model.num_layers
intermediate_size = infer_model.intermediate_size
over_zeros = torch.zeros(num_layers, intermediate_size, dtype=torch.int32)
over_zeros_dict = {"lang":0,"num" : 0, "over_zero" : torch.tensor([])}


def save_raw_vals_to_dict(name, dictname, savees):
    if not name in dictname:
        dictname[name] = savees
    else:
        dictname[name] = torch.cat([dictname[name], savees], dim=0)
    
def clean_hooks(infer_model):
    for i in range(len(infer_model.model.model.layers)):
        mlp = infer_model.model.model.layers[i].mlp
        mlp.act_fn._forward_hooks.clear()

def get_activation_mlp(name, id_prompt_start, id_prompt_end, sequence_length, max_tokens_overzeros = 10000): 
    """
        name (str): buat namain layer
        is_averaged_tokens (boolean): true if avergaed across tokens, else get the last token
    """
    def hook_fn(module, input, output):
        global raw_values_avg_tokens
        # global raw_values_last_token
        global over_zeros
        global over_zeros_dict

        # print(f"output shape {output.shape}")
        assert output.size(1) == sequence_length, f"Mismatch: output.size(1) = {output.size(1)}, sequence_length = {sequence_length}"
        
        number_of_tokens = id_prompt_end-id_prompt_start
        # print(f"number of tokens: {number_of_tokens}")
        if over_zeros_dict['num'] + number_of_tokens <= max_tokens_overzeros:
            over_zero = (output[:,id_prompt_start:id_prompt_end,:] > 0).detach().sum(dim=(0,1)).to(dtype=torch.int32).cpu()
            over_zeros[int(name), :] += over_zero
            # print(f"a: {(output[:,id_prompt_start:id_prompt_end,:] > 0).detach().sum(dim=(0,1)).to(dtype=torch.int32).cpu()}, shape: {output[:, start:end, :].shape}")
            # cc = output[:,id_prompt_start:id_prompt_end,:].detach().cpu()
            # print(f"a: {cc}, shape: {cc.shape}")
            # nonbatched.append(output[0,id_prompt_start:id_prompt_end,:] )
            # over_zeros_dict['num'] += number_of_tokens
            over_zeros_dict['over_zero'] = over_zeros
            tensor = (output[0,id_prompt_start:id_prompt_end,:] > 0).detach().to(dtype=torch.int32).cpu()
            tambahan = (output[0,id_prompt_start:id_prompt_end,:] > 0).detach().sum(dim=(0,1)).to(dtype=torch.int32).cpu()
            # print(f"y: {tensor}")
            # print(f"tambahan: {tambahan}")
            # print(f"over_zeros_dict: {over_zeros_dict}")
            
            # print(f"over_zeros.shape: {over_zeros.shape}")
        
        avg_output = output[:,id_prompt_start:id_prompt_end,:].detach().half().mean(dim=1).cpu() 
        # print(f"avg khusus {avg_output.shape}")
        # print(f"avg_output.shape: {avg_output.shape}")
        save_raw_vals_to_dict(name, raw_values_avg_tokens, avg_output)
        # else:
        # last_output = output.half()[:, -1,:].cpu() # simpen the last token
        # # print(f"last_output.shape: {last_output.shape}")
        # save_raw_vals_to_dict(name, raw_values_last_token, last_output)
        # print(f"ini raw_values_avg_token {raw_values_avg_tokens}")
        # print(len(raw_values_last_token.values()))
    return hook_fn

def concat_neuron_layers(raw_values_avg_tokens):
    full_raw_values_avg_tokens = list(raw_values_avg_tokens.values())[0]
    
    
    # print(f"full_raw_values_avg_tokens.shape {full_raw_values_avg_tokens.shape}")
    # print(full_raw_values_last_token.shape)
    for i in list(raw_values_avg_tokens.values())[1:]:
        # print(i.shape)
        full_raw_values_avg_tokens = torch.cat([full_raw_values_avg_tokens, i], dim=-1).cpu()
    # for i in list(raw_values_last_token.values())[1:]:
    #     full_raw_values_last_token = torch.cat([full_raw_values_last_token, i], dim=-1).cpu()
    # # print(f"shape full_raw_values_last_token {full_raw_values_last_token.shape}")
    return full_raw_values_avg_tokens


# def merge_avg_last(full_raw_values_avg_tokens, full_raw_values_last_token):
#     # print(f"shape full_raw_values_last_token {full_raw_values_last_token.shape}")
#     full_raw_values = torch.stack((full_raw_values_avg_tokens, full_raw_values_last_token), dim=0).cpu()
#     return full_raw_values


def register_hook(infer_model, handlers, id_prompt_start, id_prompt_end, sequence_length, max_tokens_overzeros=10000): 
    # remove_hooks(handlers)  # Remove any existing hooks before adding new ones
    clean_hooks(infer_model)
    num_layers = infer_model.num_layers
    remove_hooks(handlers)
    for i in range (num_layers):
        mlp = infer_model.model.model.layers[i].mlp
        handlers.append(mlp.act_fn.register_forward_hook(get_activation_mlp(f"{i}", id_prompt_start, id_prompt_end, sequence_length, max_tokens_overzeros)))
    # print(infer_model.model.model.layers[1].mlp.act_fn._forward_hooks)

    # for handler in handlers:
    #     handler.remove()
    # return handlers


def remove_hooks(handlers):
    for handler in handlers:
        handler.remove()
    handlers.clear()
    

def cleanup():
    raw_values_avg_tokens.clear()
    # raw_values_last_token.clear()
    full_raw_values_avg_tokens.clear()
    # full_raw_values_last_token.clear()
    # full_raw_values.clear()

def all_languages_dict_to_tensor(all_languages_dict):
    batch_size = 1000
    chunks = [torch.stack(all_languages_dict[i:i+batch_size], dim=0).cpu() 
              for i in range(0, len(all_languages_dict), batch_size)]
    full_languages_raw_values = torch.cat(chunks, dim=0)  # reassemble after stacking
    del chunks
    return full_languages_raw_values



def get_neurons(
    # models properties
    model_name,
    infer_model, 
    # dataset properties
    dataset_name, split, max_instances=None, max_lang=None, selected_langs=None, is_predict=True, apply_template=True,
    debug=False,take_whole=False,
    # neuron retrieval properties
    max_tokens_overzeros=10000,
    # save to kaggle properties
    kaggle_dataname_to_save=None, is_update=False

):

    """
    infer_model: HF model InferenceModel
    dataset_name: dataset name from HF
    split: split dataset "test", "devtest", etc
    max_instances: max_instances to be ran
    max_lang: max language to be ran. from language 0 to language max_lang. default None means all.
    selected_langs: specifiy selected_langs to be ran.
    """
    # infer_model = InferenceModel(model_name)
    global over_zeros_dict, over_zeros, raw_values_avg_tokens
    raw_values_avg_tokens = {}
    # raw_values_last_token = {}
    full_raw_values_avg_tokens = []
    # full_raw_values_last_token = []
    # full_raw_values = []

    
    eval_result = {}
    all_languages = []
    all_languages_over_zero = []
    handlers = []
    configs = get_dataset_config_names(dataset_name, trust_remote_code=True)
    n_instances = 0
    n_lang = 0
    language_dict = {}
    selected_langs = selected_langs if selected_langs != None else configs
    print(selected_langs)
    
    for lang in selected_langs:
        if lang.startswith("all"):
            continue
        if max_lang and n_lang >= max_lang:
            break
        cleanup()
        over_zeros_dict["lang"] = lang
        language_dict[n_lang] = lang 
        ds = load_dataset(dataset_name, lang, split=split, trust_remote_code=True, cache_dir=f"{parent_dir}/data/hf_datasets")
        dataset_instance = dataset_instance = Dataset(dataset_name, ds)
        if dataset_name.endswith("massive"):
            dataset_instance = Massive(dataset_name, ds)
        elif dataset_name.endswith("xcopa"): 
            dataset_instance = Xcopa(dataset_name, ds)
        elif dataset_name.endswith("xwinograd"):
            dataset_instance = Xwinograd(dataset_name, ds)
        elif dataset_name.endswith("flores200"):
            dataset_instance = Flores(dataset_name, ds)
        elif dataset_name.endswith("MLAMA-dod-185"):
            dataset_instance = Mlama(dataset_name, ds)
        else:
            raise ValueError("Dataset is not available!")
        dataset_name = dataset_instance.dataset_name
        for data in tqdm(ds, desc=f"Processing {lang} Examples", leave=False):
            if max_instances and n_instances >= max_instances:
                break
            detail_data = dataset_instance.get_detail_inference(data)
            # print(f"detail_data: {detail_data}")
            id_prompt_start, id_prompt_end, len_sentence, prompt_whole = dataset_instance.get_index_start_end_prompt(infer_model, detail_data, infer_model.model_name, is_predict, take_whole)
            clean_hooks(infer_model)
            number_of_tokens = id_prompt_end-id_prompt_start
            # print(f"number of tokens: {number_of_tokens}")
            
            register_hook(infer_model, handlers, id_prompt_start, id_prompt_end, len_sentence, max_tokens_overzeros)
            text = infer_model.get_templated_prompt(prompt_whole, apply_template)
            
            generated_text, len_sentence_model = infer_model.inference(text, debug=debug)
            if over_zeros_dict['num'] + number_of_tokens <= max_tokens_overzeros: 
                over_zeros_dict['num'] += number_of_tokens
            assert len_sentence == len_sentence_model, f"Mismatch len sentence model {len_sentence_model} and prompt {len_sentence}"
            clean_hooks(infer_model)
            remove_hooks(handlers)
            n_instances += 1
            # break
            if debug:
                print(f"text: {text}")
                print(f"generated_text: {generated_text}")               
                # print(f"prompt_whole: {prompt_whole} \n size: {infer_model.tokenizer([prompt_whole], return_tensors='pt', add_special_tokens=False).to(infer_model.model.device)['input_ids'].shape}")
                model_inputs = infer_model.tokenizer([text], return_tensors="pt", add_special_tokens=False).to(infer_model.model.device)
                print(f"(model_inputs['input_ids'] {(model_inputs['input_ids'])}")
                # print(f"len(model_inputs['input_ids'] {len(model_inputs['input_ids'])}")
                sentence_token_texts = infer_model.tokenizer.convert_ids_to_tokens((model_inputs['input_ids'])[0])
                print(f"with prompt: {sentence_token_texts}")
                print(f"without prompt: {sentence_token_texts[id_prompt_start:id_prompt_end]}\n shape:{len(sentence_token_texts[id_prompt_start:id_prompt_end])}")
                
        
            # print(f"🔵 After inference, raw_values_avg_tokens keys: {list(raw_values_avg_tokens.keys())}")
        # print(f"🔵 After inference language, raw_values_avg_tokens: {list(raw_values_avg_tokens.values())}")
        full_raw_values_avg_tokens = concat_neuron_layers(raw_values_avg_tokens)
        # full_raw_values = merge_avg_last(full_raw_values_avg_tokens, full_raw_values_last_token)
        all_languages.append(full_raw_values_avg_tokens)
        all_languages_over_zero.append(over_zeros_dict)
        print()
        over_zeros = torch.zeros(infer_model.num_layers, infer_model.intermediate_size, dtype=torch.int32)
        over_zeros_dict = {"lang":0,"num" : 0, "over_zero" : torch.tensor([])}
        # print(full_raw_values.shape)
        n_lang += 1
        print(f"Done with lang: {lang}")
        for handler in handlers:
            handler.remove()
        # if n_lang >= 2: break # 
        n_instances = 0
        # break
        # print(infer_model.model.model.layers[1].mlp.act_fn._forward_hooks)
    cleanup()
    del raw_values_avg_tokens
    # del raw_values_last_token
    del full_raw_values_avg_tokens
    # del full_raw_values_last_token
    # del full_raw_values 
    print(f"all_languages: {len(all_languages)}")
    full_languages_raw_values = all_languages_dict_to_tensor(all_languages)

    path_res = f"{parent_dir}res/act/{model_name.split('/')[-1]}"
    os.makedirs(path_res, exist_ok=True)
    torch.save(full_languages_raw_values, f"{path_res}/act_{dataset_name.split('/')[-1]}_{max_instances}_{is_predict}.pt")
    torch.save(all_languages_over_zero, f"{path_res}/oz_{dataset_name.split('/')[-1]}_{max_instances}_{is_predict}")
    torch.save(language_dict, f"{path_res}/ld_{dataset_name.split('/')[-1]}")

    if kaggle_dataname_to_save:
        save_to_kaggle(dataset_name= kaggle_dataname_to_save, data_dir=path_res, is_update=is_update)

    return full_languages_raw_values, all_languages_over_zero, language_dict

d, e, f = get_neurons(
        model_name=args.model,
        infer_model=infer_model,
        dataset_name=args.dataset_name,
        split=args.split,
        max_instances=args.max_instances,
        max_lang=args.max_lang,
        selected_langs=args.selected_langs,
        is_predict=args.is_predict,
        apply_template=args.apply_template,
        debug=args.debug,
        take_whole=args.take_whole,
        max_tokens_overzeros=args.max_tokens_overzeros,
        kaggle_dataname_to_save=args.kaggle_dataname_to_save, 
        is_update=args.is_update
    )