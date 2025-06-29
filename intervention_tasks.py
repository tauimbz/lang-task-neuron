import argparse
from argparse import Namespace
from huggingface_hub import login
from datasets import load_dataset, get_dataset_config_names
from model import InferenceModel
import torch.nn.functional as F
import math
import torch 
from tqdm import tqdm
from IPython.display import display
from kaggle_utils import *
import evaluate
import numpy as np
import xcopa_utils
import xwinograd_utils

def generate(model, prompt, with_template=True, max_new_tokens=1):
    """
    Performs inference on a given prompt.
    Returns the decoded output.
    This method only aims to generate answer. No prior answer needed.
    Input: model, prompt,
    Output: generated_text, len_sentence
    """
    generated_text = ''
    len_sentence = 0
    text = model.get_templated_prompt(prompt, with_template)
    model_inputs = model.tokenizer([text], return_tensors="pt").to(model.model.device)
    generated_ids = model.model.generate(
            **model_inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=max_new_tokens
        )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    generated_text = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    len_sentence = len(model.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)["input_ids"])
    return generated_text, len_sentence



def calc_logprob_answeroptions(model, sentences, target_spans):
    """
    Compute log probabilities of the target spans dynamically inside each sentence.

    Parameters:
    - model: The language model
    - tokenizer: Tokenizer for processing input text
    - sentences: List of full sentences
    - target_spans: List of the expected target spans

    Returns:
    - Dictionary mapping each detected target span to its summed log probability.
    """
    inputs = model.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

    log_probs = torch.log_softmax(logits, dim=-1)
    # print(f"log_probs: {log_probs}, {len(log_probs)}, shape {log_probs[0].shape}")

    # print(f"input_ids: {input_ids}")
    
    batch_seq_log_probs = {}
    
    for i in range(input_ids.size(0)):  # Loop over batch
        seq_log_prob = 0
        for j in range(1, input_ids.size(1)):  # Ignore first token (predict from previous)
            token_id = input_ids[i, j]
            if attention_mask[i, j] == 1:  # Ignore padding tokens
                seq_log_prob += log_probs[i, j-1, token_id].item()
        batch_seq_log_probs[target_spans[i]] = seq_log_prob
        # batch_seq_log_probs.append(seq_log_prob)
    
    return batch_seq_log_probs      


def eval(predictions, references, metrics = None):
    """
    used for an entire dataset (per language).
    metrics: list of metrics being used: ['acc', 'prec', 'recall'], None means use all. 
    Evaluates the predictions against the references 
    """
    acc_metric = evaluate.load("accuracy")  
    precision_metric = evaluate.load("precision")  
    recall_metric = evaluate.load("recall") 
    accuracy_results = acc_metric.compute(predictions=predictions, references=references)

    precision_results = precision_metric.compute(predictions=predictions, references=references, average="macro")

    recall_results = recall_metric.compute(predictions=predictions, references=references, average="macro")

    results = {
        'acc' : accuracy_results['accuracy'],
        'prec' : precision_results['precision'],
        'recall' : recall_results['recall']
    }
    results = results if metrics is None else {k: v for k, v in results.items() if k in metrics}
    return results

def eval_ppl(perplexities):
    avg_perplexity = np.mean(perplexities)
    return avg_perplexity
def eval_dod(preds, refs):
    assert len(preds) == len(refs)
    # print(f"preds: {preds}")
    # print(f"refs: {refs}")
    
    diff_of_diff = [a - b for a, b in zip(preds, refs)]
    count_dod = sum(1 for diff in diff_of_diff if diff > 0)
    res_dod = count_dod/len(preds)
    return res_dod


def clean_hooks(infer_model):
    for i in range(len(infer_model.model.model.layers)):
        mlp = infer_model.model.model.layers[i].mlp
        mlp.act_fn._forward_hooks.clear()
def set_activation_mlp_v2(replace_method, replacer_tensor, model_name, name, lsn_langs, target_lang, operation_non_target, operation_target, start_idx=None, attn_mask=None): 
    """
    This changes all neuron lape values to be replaced_values but leave behind a desired target language. 
        replace_method: lape or all
        name (str): buat namain layer
        lsn_langs: dict semua language lapef
        target_lang: int index target_lang in lape. Has to exist in the lsn_langs and have the same index.
        operation: *10, =0, +5, etc
    """
    operand_t = operation_target[0]
    replace_value_t = int(operation_target[1:])
    if operand_t not in ["=", "*", "+"]:
        raise ValueError("operand is wrong!")

    operand_nt = operation_non_target[0]
    replace_value_nt = int(operation_non_target[1:])
    if operand_nt not in ["=", "*", "+", "."]:
        raise ValueError("operand is wrong!")
    # print(f"replace_value_t: {replace_value_t}")
    def hook_fn(module, input, output):
        # print(f"output {output.shape}, target_lang={lsn_languages[target_lang]}")
        # start_id_to_intv = -3 if model_name == "Qwen/Qwen2.5-0.5B-Instruct" or model_name == 'Qwen/Qwen2.5-3B-Instruct' else 0
        start_id_to_intv = 0
        layer = int(name)
        if replacer_tensor is not None:
            lsn_lang = lsn_langs[target_lang]
            if lsn_lang[layer].numel() == 0:
                return
            # output[:, :, lsn_lang[layer].long()] = replacer_tensor[target_lang][layer][lsn_lang[layer]].to(output.dtype)
            dims = lsn_lang[layer].long()  # [H']
            replacement_values = replacer_tensor[target_lang][layer][dims].to(output.dtype)  # [H']
            mask = attn_mask.unsqueeze(-1)  # [B, T, 1]
            
            output_selected = output[:, :, dims]  # [B, T, H']
            
            # [H'] -> [1, 1, H'] so it can broadcast across B and T
            replacement_broadcasted = replacement_values.view(1, 1, -1)
            
            output[:, :, dims] = torch.where(
                mask.bool(),  # [B, T, 1]
                replacement_broadcasted.expand_as(output_selected),  # [B, T, H']
                output_selected  # keep original where mask is 0
            )
        else:
            for lang, lsn_lang in lsn_langs.items():
                # print("ini intervensi")
                if lsn_lang[layer].numel() == 0:
                    continue
                if replace_method == "fixed":
                    indexing_tensor = lsn_lang[layer].long().to(output.device)
                    if indexing_tensor.numel() == 0:
                        continue
                    assert indexing_tensor.max() < output.shape[-1], "Index out of bounds 2!"
                    if lang == target_lang:
                        if operand_t == "*":
                            output[0, start_id_to_intv:, lsn_lang[layer].long()] *= replace_value_t
                        elif operand_t == "=":
                            output[0, start_id_to_intv:, lsn_lang[layer].long()] = replace_value_t
                        else:
                            output[0, start_id_to_intv:, lsn_lang[layer]] += replace_value_t
                    else:
                        if operand_nt == "*":
                            output[0, start_id_to_intv:, lsn_lang[layer]] *= replace_value_nt
                        elif operand_nt == "=":
                            output[0, start_id_to_intv:, lsn_lang[layer].long()] = replace_value_nt
                        elif operand_nt == ".":
                            pass
                        else:
                            output[0, start_id_to_intv:, lsn_lang[layer].long()] += replace_value_nt
                elif replace_method == "mean":
                    selected = output[0, start_id_to_intv:, lsn_lang[layer]]
                    if selected.numel() == 0:
                        continue
                    mean_value = output.mean()
                    # print(output[0, start_id_to_intv:, lsn_lang[layer]].numel())
                    
                    if lang == target_lang:
                        if operand_t == "*":
                            output[0, start_id_to_intv:, lsn_lang[layer]] *= mean_value * replace_value_t
                        elif operand_t == "=":
                            output[0, start_id_to_intv:, lsn_lang[layer]] = mean_value * replace_value_t
                        else:
                            # TODO: tadinya output[0, start_id_to_intv:, lsn_lang[layer]] += abs(mean_value * replace_value_t)
                            output[0, start_id_to_intv:, lsn_lang[layer]] += (mean_value * replace_value_t)
                    else:
                        if operand_nt == "*":
                            output[0, start_id_to_intv:, lsn_lang[layer]] *= mean_value * replace_value_nt
                        elif operand_nt == "=":
                            output[0, start_id_to_intv:, lsn_lang[layer]] = mean_value * replace_value_nt
                        elif operand_nt == ".":
                            output[0, start_id_to_intv:, lsn_lang[layer]] *= 1
                        else:
                            output[0, start_id_to_intv:, lsn_lang[layer]] += mean_value * replace_value_nt
        
                elif replace_method == "std":
                    selected = output[0, start_id_to_intv:, lsn_lang[layer]]
                    if selected.numel() == 0:
                        continue
                    mean_value = output.std()
                    
                    if lang == target_lang:
                        if operand_t == "*":
                            output[0, start_id_to_intv:, lsn_lang[layer]] *= mean_value * replace_value_t
                        elif operand_t == "=":
                            output[0, start_id_to_intv:, lsn_lang[layer]] = mean_value * replace_value_t
                        else:
                            # TODO: tadinya output[0, start_id_to_intv:, lsn_lang[layer]] += abs(mean_value * replace_value_t)
                            output[0, start_id_to_intv:, lsn_lang[layer]] += (mean_value * replace_value_t)
                    else:
                        if operand_nt == "*":
                            output[0, start_id_to_intv:, lsn_lang[layer]] *= mean_value * replace_value_nt
                        elif operand_nt == "=":
                            output[0, start_id_to_intv:, lsn_lang[layer]] = mean_value * replace_value_nt
                        elif operand_nt == ".":
                            output[0, start_id_to_intv:, lsn_lang[layer]] *= 1
                        else:
                            output[0, start_id_to_intv:, lsn_lang[layer]] += mean_value * replace_value_nt
                elif replace_method == "iqr":
                    selected = output[0, start_id_to_intv:, lsn_lang[layer]]
                    if selected.numel() == 0:
                        continue
                    q1 = output.quantile(0.25)
                    q3 = output.quantile(0.75)
                    mean_value = q3 - q1
                    
                    if lang == target_lang:
                        if operand_t == "*":
                            output[0, start_id_to_intv:, lsn_lang[layer]] *= mean_value * replace_value_t
                        elif operand_t == "=":
                            output[0, start_id_to_intv:, lsn_lang[layer]] = mean_value * replace_value_t
                        else:
                            output[0, start_id_to_intv:, lsn_lang[layer]] += (mean_value * replace_value_t)
                    else:
                        if operand_nt == "*":
                            output[0, start_id_to_intv:, lsn_lang[layer]] *= mean_value * replace_value_nt
                        elif operand_nt == "=":
                            output[0, start_id_to_intv:, lsn_lang[layer]] = mean_value * replace_value_nt
                        elif operand_nt == ".":
                            output[0, start_id_to_intv:, lsn_lang[layer]] *= 1
                        else:
                            output[0, start_id_to_intv:, lsn_lang[layer]] += mean_value * replace_value_nt
                elif replace_method == "mad":
                    selected = output[0, start_id_to_intv:, lsn_lang[layer]]
                    if selected.numel() == 0:
                        continue
                    median = output.median()
                    mean_value = (selected - median).abs().median()
                    # print(output[0, start_id_to_intv:, lsn_lang[layer]].numel())
                    
                    if lang == target_lang:
                        if operand_t == "*":
                            output[0, start_id_to_intv:, lsn_lang[layer]] *= mean_value * replace_value_t
                        elif operand_t == "=":
                            output[0, start_id_to_intv:, lsn_lang[layer]] = mean_value * replace_value_t
                        else:
                            # TODO: tadinya output[0, start_id_to_intv:, lsn_lang[layer]] += abs(mean_value * replace_value_t)
                            output[0, start_id_to_intv:, lsn_lang[layer]] += (mean_value * replace_value_t)
                    else:
                        if operand_nt == "*":
                            output[0, start_id_to_intv:, lsn_lang[layer]] *= mean_value * replace_value_nt
                        elif operand_nt == "=":
                            output[0, start_id_to_intv:, lsn_lang[layer]] = mean_value * replace_value_nt
                        elif operand_nt == ".":
                            output[0, start_id_to_intv:, lsn_lang[layer]] *= 1
                        else:
                            output[0, start_id_to_intv:, lsn_lang[layer]] += mean_value * replace_value_nt
                elif replace_method == "median":
                    selected = output[0, start_id_to_intv:, lsn_lang[layer]]
                    if selected.numel() == 0:
                        continue
                    mean_value = output.median()
                    # print(output[0, start_id_to_intv:, lsn_lang[layer]].numel())
                    
                    if lang == target_lang:
                        if operand_t == "*":
                            output[0, start_id_to_intv:, lsn_lang[layer]] *= mean_value * replace_value_t
                        elif operand_t == "=":
                            output[0, start_id_to_intv:, lsn_lang[layer]] = mean_value * replace_value_t
                        else:
                            # TODO: tadinya output[0, start_id_to_intv:, lsn_lang[layer]] += abs(mean_value * replace_value_t)
                            output[0, start_id_to_intv:, lsn_lang[layer]] += (mean_value * replace_value_t)
                    else:
                        if operand_nt == "*":
                            output[0, start_id_to_intv:, lsn_lang[layer]] *= mean_value * replace_value_nt
                        elif operand_nt == "=":
                            output[0, start_id_to_intv:, lsn_lang[layer]] = mean_value * replace_value_nt
                        elif operand_nt == ".":
                            output[0, start_id_to_intv:, lsn_lang[layer]] *= 1
                        else:
                            output[0, start_id_to_intv:, lsn_lang[layer]] += mean_value * replace_value_nt
                
                elif replace_method == "percent": 
                    
                    dims = lsn_lang[layer]  # [H']
                    if dims.numel() == 0:
                        continue

                    for b in range(output.size(0)):  # Loop over batch
                        mask_b = attn_mask[b]  # [T]
                        valid_idx = mask_b.bool().nonzero(as_tuple=False).squeeze(1)  # [#valid_T]

                        if valid_idx.numel() == 0:
                            continue

                        # Get only valid time steps and selected dims
                        selected = output[b, valid_idx, :]  # [#valid_T, H']

                        if selected.numel() == 0:
                            continue

                        flat = selected.flatten()
                        k = max(int(flat.numel() * (replace_value_t / 100)), 1)
                        percentile_val = flat.kthvalue(k).values

                        if lang == target_lang:
                            if operand_t == "*":
                                output[b, valid_idx[:, None], dims] *= percentile_val
                            elif operand_t == "=":
                                output[b, valid_idx[:, None], dims] = percentile_val
                            else:
                                output[b, valid_idx[:, None], dims] += percentile_val
                        else:
                            if operand_nt == "*":
                                output[b, valid_idx[:, None], dims] *= percentile_val
                            elif operand_nt == "=":
                                output[b, valid_idx[:, None], dims] = percentile_val
                            elif operand_nt == ".":
                                pass
                            else:
                                output[b, valid_idx[:, None], dims] += percentile_val
    
                elif replace_method == "max":
                    # print(f"max: ")
                    # print(f"len()lsn_lang: {len(lsn_lang)}")
                    # print(f"len()lsn_lang: {len(lsn_lang)}")
                    # print(f"layer: {layer}")
                    if lsn_lang[layer].numel() == 0:
                        continue
                    
                    selected = output[0, start_id_to_intv:, lsn_lang[layer]]
                    # print(f"selected {selected}")
                    if selected.numel() == 0:
                        continue
                    mean_value = output.max()
                    # print(f"max: {mean_value}")
                    # print(output[0, start_id_to_intv:, lsn_lang[layer]].numel())
                    
                    if lang == target_lang:
                        if operand_t == "*":
                            output[0, start_id_to_intv:, lsn_lang[layer]] *= mean_value * replace_value_t
                        elif operand_t == "=":
                            output[0, start_id_to_intv:, lsn_lang[layer]] = mean_value * replace_value_t
                        else:
                            # TODO: tadinya output[0, start_id_to_intv:, lsn_lang[layer]] += abs(mean_value * replace_value_t)
                            output[0, start_id_to_intv:, lsn_lang[layer]] += (mean_value * replace_value_t)
                    else:
                        if operand_nt == "*":
                            output[0, start_id_to_intv:, lsn_lang[layer]] *= mean_value * replace_value_nt
                        elif operand_nt == "=":
                            output[0, start_id_to_intv:, lsn_lang[layer]] = mean_value * replace_value_nt
                        elif operand_nt == ".":
                            output[0, start_id_to_intv:, lsn_lang[layer]] *= 1
                        else:
                            output[0, start_id_to_intv:, lsn_lang[layer]] += mean_value * replace_value_nt
        
            
    return hook_fn
    


def calculate_logprob(model, prompt: str, continuation: str, is_generate=False ) -> float:
    full_input = prompt + " " + continuation if continuation else prompt
    inputs = model.tokenizer(full_input, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.model(**inputs)
    
    logits = outputs.logits[:, :-1, :]  # [batch, seq_len-1, vocab_size]
    labels = inputs.input_ids[:, 1:]    # [batch, seq_len-1]

    # only want to score the continuation part
    continuation_labels = labels[:, :]
    continuation_logits = logits[:, :]
    if continuation:
        prompt_len = len(model.tokenizer(prompt, return_tensors="pt").input_ids[0])
        continuation_labels = labels[:, prompt_len - 1:]  # trim prompt tokens
        continuation_logits = logits[:, prompt_len - 1:]
    
    # Gather log probs of actual tokens
    log_probs = torch.nn.functional.log_softmax(continuation_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, continuation_labels.unsqueeze(-1)).squeeze(-1)

    total_log_prob = token_log_probs.sum().item()
    if is_generate:
        print(f"prompt: {prompt}")
        token_log_probs = token_log_probs.squeeze().tolist()
        for i, logp in enumerate(token_log_probs):
            print(f"Token {i}: log_prob = {logp:.4f}, perplexity = {math.exp(-logp):.4f}")
        print(f"log_prob: {total_log_prob}")
    # total_log_prob = np.array(total_log_prob).reshape(len(prompt), 4)
    return total_log_prob

def tokenize_batch(model, prompts: list[str], continuations: list[str]):
    assert len(prompts) == len(continuations), "Mismatch between prompts and continuations"
    inputs = [p + " " + c if c else p for p, c in zip(prompts, continuations)]

    # Tokenize all inputs with padding
    encoding = model.tokenizer(inputs, return_tensors="pt", padding=True)
    input_ids = encoding["input_ids"].to(model.device)
    attention_mask = encoding["attention_mask"].to(model.device)
    return input_ids, attention_mask
    
def calculate_logprob_batch(model, input_ids, attention_mask, prompts: list[str], continuations: list[str]) -> list[float]:
    """
    compute log-probs for a batch of (prompt, continuation) pairs.
    returns a list of total log-probs, one per input.
    """
    model.model.eval()
    with torch.no_grad():
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # shift for causal LM
        labels = input_ids[:, 1:]           # shift labels accordingly

    # Compute log-probs
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather log probs of actual tokens
    token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)

    # Mask padding
    loss_mask = (labels != model.tokenizer.pad_token_id)
    
    # Compute total logprob only over continuation part
    log_probs_per_example = []
    for i, (prompt, continuation) in enumerate(zip(prompts, continuations)):
        prompt_len = len(model.tokenizer(prompt, return_tensors="pt").input_ids[0]) - 1
        cont_log_probs = token_log_probs[i, prompt_len:]
        cont_mask = loss_mask[i, prompt_len:]
        total = cont_log_probs[cont_mask].sum().item()
        log_probs_per_example.append(total)
    # del logits, log_probs
    return log_probs_per_example


def calc_perplexity_answer(eval_type, prompt, continuation, model, is_generate=False):
    perplexity = 0
    if eval_type == "EVAL_PPL_FULL":
        sentence = prompt + " " + continuation
        inputs = model.tokenizer(sentence, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids
    
        with torch.no_grad():
            outputs = model.model(**inputs, labels=input_ids)
            loss = outputs.loss  # mean NLL over all tokens
        perplexity = torch.exp(loss).item()
    else:
        log_prob = calculate_logprob(model, prompt, continuation)
        num_tokens = len(model.tokenizer(continuation, return_tensors="pt").input_ids[0])
        perplexity = math.exp(-log_prob / num_tokens)
        # print("perplexity:", perplexity)
    if is_generate:
        print(f"perplexity: {perplexity}")
    return perplexity

def calc_perplexity_batch(
    eval_type: str,
    prompts: List[str],
    continuations: List[str],
    model,
    is_generate: bool = False
) -> List[float]:
    perplexities = []

    if eval_type == "EVAL_PPL_FULL":
        sentences = [p + " " + c for p, c in zip(prompts, continuations)]
        inputs = model.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(model.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        with torch.no_grad():
            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss  # Mean loss across all tokens and batch
            # Need per-example loss, so we manually calculate it below

            # Compute per-token loss
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            per_token_loss = per_token_loss.view(shift_labels.size())

            # Sum losses and divide by number of tokens per example
            token_sums = per_token_loss.sum(dim=1)
            token_counts = shift_mask.sum(dim=1)
            ppl_per_example = torch.exp(token_sums / token_counts)

            perplexities = ppl_per_example.tolist()

    else:
        # Unbatched fallback for non-EVAL_PPL_FULL modes
        for prompt, continuation in zip(prompts, continuations):
            log_prob = calculate_logprob(model, prompt, continuation)
            num_tokens = len(model.tokenizer(continuation, return_tensors="pt").input_ids[0])
            perplexity = math.exp(-log_prob / num_tokens)
            perplexities.append(perplexity)

    if is_generate:
        for i, ppl in enumerate(perplexities):
            print(f"[{i}] perplexity: {ppl:.4f}")
    return perplexities


def HF_calculate_answer(ds, data, dataset_name, model, eval_type, is_generate, dod_baselang= None, dod_languages=None):
    """
    Input:
    - ds: return of load_dataset(), get special configuration for some dataset e.g intents_set. 
        may not be used for other datasets
    - dataset_name: dataset name
    - model: InferenceModel instance
    - text: prompt with applied (or not) template, with special token. output of model.get_templated_prompt
    - dod_languages: is languages to be intervened with dod
    Output:
    - dod_baselang: str, the base language
    - option_log_probs (list of tuple of (option, its seq_log_prob)), answer_log_probs (tuple of (option, its seq_log_prob))
    """
    if (eval_type.startswith("DOD") and not dod_baselang):
        raise ValueError("DOD props must be provided")
        
    choices = []
    gold = ""
    premise = ""
    correct_sentence = ""
    correct_idx = None
    target = None
    num_choices = None
    if dataset_name == "AmazonScience/massive":
        # ds = load_dataset(dataset_name, lang, split=split)
        intents = ds.features['intent'].names
        scenarios = ds.features['scenario'].names
        scenario = scenarios[data['scenario']]
        gold = intents[data['intent']].split("_", 1)[-1]
        choices = [intent[intent.find('_')+1:] for intent in intents if intent.startswith(scenario)]
        premise = data["utt"]
        correct_sentence = f"The intent of {premise} is {gold}"
        
    elif dataset_name == "facebook/xnli":
        choices = ['entailment', 'neutral', 'contradiction']
        gold = choices[data['label']]
        premise = data["premise"]
        correct_sentence = f"The relation between the premise and  hypothesis is {gold}"
    elif dataset_name == "Muennighoff/xwinograd": #👍 same as lm eval harness
        option1 = data['option1']
        option2 = data['option2']
        choices = xwinograd_utils.doc_to_choice(data)
        target = xwinograd_utils.doc_to_target(data)
        correct_idx = xwinograd_utils.doc_to_text(data)
        gold = choices[correct_idx]
        correct_sentence = gold + " " + target
        num_choices = 2
        target = [target for i in range(num_choices)]
        
    elif dataset_name == "CohereLabs/include-lite-44": #👍 same as lm eval harness
        option_a, option_b, option_c, option_d = tuple(data['choices'])
        question = data['question']
        correct_idx = data['answer']
        target = ['A', 'B', 'C', 'D'] 
        choices = f"{question.strip()}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nAnswer:"
        gold = target[correct_idx]
        correct_sentence = choices + " " + gold
        num_choices = 4
        choices = [choices for i in range(num_choices)]
        
    elif dataset_name.endswith("MLAMA-dod-185"):
        # print(f"dod_languages: {dod_languages}")
        base_lang_sub = f'sub_{dod_baselang}'
        base_lang_obj = f'obj_{dod_baselang}'
        dod_lang_dict = {k:v for k, v in enumerate(dod_languages)}
        dod_lang_idx = {v:k for k,v in dod_lang_dict.items()}
        
        dod_all_langs = [i for i in dod_languages]
        
        choices.extend([data[f"obj_{i}"] for i in dod_all_langs])
        # print(f"dod_baselang: {dod_baselang}")
        premise = data['template'].replace("[X]", data[base_lang_sub]).replace(" [Y]", "")
        # print(f"premise: {premise}")
        correct_idx = dod_lang_idx[dod_baselang]
        gold = choices[correct_idx]
        correct_sentence = premise + " " + gold
    elif dataset_name == "cambridgeltl/xcopa":
        choice1 = data['choice1']
        choice2 = data['choice2']
        question = data['question']
        answer_options = [choice1, choice2]

        fn_name = f"doc_to_text_{dod_baselang}"
        doc_to_text_lang = getattr(xcopa_utils, fn_name)
        correct_idx = data['label']
        choices = doc_to_text_lang(data)
        target = xcopa_utils.doc_to_choice(data)
        gold = target[correct_idx]
        correct_sentence = choices + " " + gold
        num_choices = 2
        choices = [choices for i in range(num_choices)]
    elif dataset_name == "facebook/flores":
        print(f"data.keys: {data.keys()}")
        base_lang_sentence = 'sentence_eng'
        choice1 = [i for i in data.keys() if i.startswith(base_lang_sentence)]
        choice2 = []
        if len(choice1) == 0:
            choice1 = [i for i in data.keys() if i.startswith("sentence")][0]
            choice2 = [i for i in data.keys() if i.startswith("sentence")][0]
        else:
            choice1 = choice1[0]
            choice2 = [i for i in data.keys() if i.startswith("sentence") and i != choice1][0]
        
        choice1 = data[choice1]
        choice2 = data[choice2]
        gold = choice2
        # choice1 = data['choice1']
        # choice2 = data['choice2']
        # question = data['question']
        choices = [choice1, choice2]
        target = ""
        correct_sentence = choice2
        correct_idx = 1
    else:
        print("Dataset is not available yet!")
        raise ValueError("Dataset is not available yet!")
    assert len(choices) != 0
    assert gold
    assert correct_idx != None
    
    chosen_prob = float('-inf')
    option_log_probs = []
    pred_log_prob = tuple()
    gold_log_prob = tuple()
    if eval_type == "DOD_NINT":
        assert dod_lang_dict, "dod_lang_dict should be defined"
        # difflogprob_per_lang = dict()
        # gold_log_prob = 0
        # print(f"dod_lang_dict: {dod_lang_dict}")
        return choices, premise, correct_idx, is_generate
        # for idx, option in enumerate(choices):
        #     logprob = calculate_logprob(model, premise, option, is_generate)
        #     if idx == correct_idx:
        #         gold_log_prob = logprob
        #         difflogprob_per_lang[dod_lang_dict[idx]] = 0
        #     else:
        #         difflogprob_per_lang[dod_lang_dict[idx]] = logprob - gold_log_prob
        # return difflogprob_per_lang
        
    if eval_type == "DOD_INT":
        assert dod_lang_dict, "dod_lang_dict should be defined"
        assert len(dod_languages) <= 2, f"more than 2 dod lang: {dod_languages}"
        assert len(choices) <= 2, f"more than 2 choices: {choices}"
        difflogprob_per_lang = dict()
        gold_log_prob = 0
        # print(f"dod_lang_dict: {dod_lang_dict}")
        difference = 0
        return choices, premise, correct_idx, is_generate
        # for idx, option in enumerate(choices):
        #     logprob = calculate_logprob(model, premise, option, is_generate)
        #     if idx == correct_idx:
        #         gold_log_prob = logprob
        #     else:
        #         difference = logprob - gold_log_prob 
        #     # seq_log_probs = calc_logprob_answer(text, option, model, is_generate)
        # return difference
            
    if eval_type == "EVAL_TASK":
        assert num_choices, "num_choices must be defined"
        return choices, target, is_generate, correct_idx, num_choices, gold
        
            
        # for idx, option in enumerate(choices):
        #     seq_log_probs = calculate_logprob(model, choices[idx], target, is_generate) 
        #     # seq_log_probs = calc_logprob_answer(text, option, model, is_generate)
        #     if option.lower() == gold.lower():
        #         gold_log_prob = (idx, seq_log_probs)
        #     option_log_probs.append((idx, option, seq_log_probs))
        #     if seq_log_probs >= chosen_prob:
        #         chosen_prob = seq_log_probs
        #         pred_log_prob = (idx, option, seq_log_probs)
        # assert len(pred_log_prob) == 3
        # assert len(gold_log_prob) == 2
        # return option_log_probs, pred_log_prob, gold_log_prob
    elif eval_type.startswith("EVAL_PPL"):
        assert correct_sentence
        return eval_type, choices[correct_idx], target, is_generate
        # perplexity_gold = calc_perplexity_answer(eval_type, choices[correct_idx], target, model, is_generate)
        # return perplexity_gold
        
def HF_make_df_per_lang(rows, prompt, eval_type, option_log_probs=None, pred_log_prob=None, gold_log_prob=None):
    row = dict()
    if eval_type == "EVAL_TASK":
        row = {
            "prompt": prompt,
            "option_logprobs": option_log_probs,
            "pred_logprobs": pred_log_prob,
            "gold_logprobs": gold_log_prob
        }  
    elif eval_type.startswith("EVAL_PPL"):
        row = {
                "prompt": prompt,
                "perplexity": gold_log_prob
            }  
    rows.append(row)
    return rows

def HF_infer_dataset(
        model, dataset_name, dataset_relations=None, langs=None, max_samples=None, is_generate=False,
        apply_template=True, batch_size=None,
        intervention = False, replace_method=None, replacer_tensor=None, lsn_langs = [], target_lang=None, operation_non_target="*1", operation_target="*1", range_layers=[1], lsn_languages={},
        split="test", show_df_per_lang=False, metrics=None, scenario=None, selected_langs = None, gold_difference = None):
    """
    model:InferenceModel
    intervention: list of languages want to be intervened
    dataset_name: dataset nane from load_dataset()
    split:train, test
    """
    handlers = []
    # if intervention:
        # # hook.intervensi_w_target_lang(model, "lape", lsn_langs, target_lang, max_new_tokens, operation_non_target, operation_target, range_layers)
        # clean_hooks(model)
        # for i in (range_layers):
        #     mlp = model.model.model.layers[i].mlp
        #     handlers.append(mlp.act_fn.register_forward_hook(
        #         set_activation_mlp_v2(
        #             replace_method=replace_method, model_name=model.model_name, name=f"{i}", lsn_langs=lsn_langs, 
        #             target_lang=target_lang, operation_non_target=operation_non_target, 
        #             operation_target=operation_target, lsn_languages=lsn_languages)))
    eval_type = ""
    if "ppl_full" in metrics :
        eval_type = "EVAL_PPL_FULL"
    elif "ppl_target" in metrics:
        eval_type = "EVAL_PPL_TARGET"
    elif 'acc' in metrics or 'prec' in metrics or 'recall' in metrics or metrics == None:
        eval_type = "EVAL_TASK"
    elif "dod" in metrics and intervention:
        eval_type = "DOD_INT"
    elif "dod" in metrics and not intervention:
        eval_type = "DOD_NINT"
    elif "bleu" in metrics:
        eval_type = "TRANSLATE"
    if eval_type == "DOD_INT" and not gold_difference:
        raise ValueError("Gold diff must be provided")
    # langs = ['eng_Latn-jpn_Jpan','eng_Latn-por_Latn', 'eng_Latn-rus_Cyrl', 'eng_Latn-zho_Hans']
    
    configs = get_dataset_config_names(dataset_name, trust_remote_code=True) if selected_langs == None else selected_langs
    selected_langs = selected_langs if selected_langs else configs
    # print(f"selected_langs: {selected_langs}")
    
    max_instances = max_samples
    dod_languages = langs if langs else selected_langs
    eval_result = {}
    non_int_dod = {}
    for lang in selected_langs:
        if lang.startswith("all"):
            continue
        result_per_lang = {
            "pred": [],
            "gold" : []
        }
        df_per_lang_rows = []
        non_int_dod[lang] = {k:[] for k in dod_languages}
        # print(f'lang: {lang}')
        
        
        
        ds = load_dataset(dataset_name, lang, split=split, trust_remote_code=True)
        if dataset_relations:
            ds = ds.filter(lambda example: example['predicate_id'] in dataset_relations)
        
        max_samples = min(max_instances, len(ds)) if max_instances else len(ds)
        
        batch_size = batch_size if batch_size else 1
        # dataset_loader = DataLoader(ds, batch_size=batch_size)
        # print(f"batch_size: {batch_size}")
        # samples = 0
        # for i, datas in enumerate(tqdm(dataset_loader, desc=f"Processing {lang} Examples", leave=False)):
        for start_idx in tqdm(range(0, max_samples, batch_size), desc=f"Processing {lang} Examples in batches", leave=False):
            # print(f"max_samples: {max_samples}")
            end_idx = min(start_idx + batch_size, max_samples)
            # print(f"processing data {start_idx} to {end_idx}")
            batch_data = ds.select(list(range(start_idx, end_idx)))
            # print(f"processing data {start_idx} to {end_idx}")
            # samples += 1
            # print(f"data: {data}")
            # print(f"Eval_type: {eval_type}")
            clean_hooks(model)
            if eval_type == "EVAL_TASK":
                batched_prompts = []
                batched_continuations = []
                batched_correct_idx = []
                num_choices = None
                # print(f"datas: {datas}")
                for data in batch_data:
                    # print(f"data: {data}")
                    choices, target, is_generate, correct_idx, num_choices, _ = HF_calculate_answer(ds, data, dataset_name, model, eval_type, is_generate=is_generate, dod_baselang=lang)
                    # print(f"choices: {choices}\ntarget: {target}")
                    assert len(choices) == len(target), "length choices and target must be the same!"
                    batched_prompts.extend(choices)
                    batched_continuations.extend(target)
                    batched_correct_idx.append(correct_idx)
                # print(f"batched_prompts: {batched_prompts}")
                # print(f"batched_continuations: {batched_continuations}")
                # print(f"batched_correct_idx: {batched_correct_idx}")
                
                assert num_choices, "num choices should not be None"
                # print(f"len(batched_prompts): {len(batched_prompts)}")
                # print(f"len(batched_continuations): {len(batched_continuations)}")
                # total_len = [
                #     len(
                #         model.tokenizer(p + " " + c, return_tensors="pt")
                #         .to(model.device)["input_ids"][0]
                #     )
                #     for p, c in zip(batched_prompts, batched_continuations)
                # ]
                # print(f"Max input length: {max(total_len)} | Avg: {sum(total_len) / len(total_len):.2f}")
                input_ids, attn_mask = tokenize_batch(model, batched_prompts, batched_continuations)
                if intervention:
                    # hook.intervensi_w_target_lang(model, "lape", lsn_langs, target_lang, max_new_tokens, operation_non_target, operation_target, range_layers)
                    clean_hooks(model)
                    for i in (range_layers):
                        mlp = model.model.model.layers[i].mlp
                        handlers.append(mlp.act_fn.register_forward_hook(
                            set_activation_mlp_v2(
                                replace_method=replace_method, replacer_tensor=replacer_tensor, model_name=model.model_name, name=f"{i}", lsn_langs=lsn_langs, 
                                target_lang=target_lang, operation_non_target=operation_non_target, 
                                operation_target=operation_target, attn_mask=attn_mask)))
                log_probs = calculate_logprob_batch(model, input_ids, attn_mask, batched_prompts, batched_continuations)
                log_probs = np.array(log_probs).reshape(len(batch_data), num_choices)
                predictions = log_probs.argmax(axis=1)
                result_per_lang['pred'].extend(predictions)
                result_per_lang['gold'].extend(batched_correct_idx)
              

            if eval_type == "TRANSLATE": #belum selesai sampai sini
                batched_prompts = []
                batched_continuations = []
                batched_correct_idx = []
                num_choices = None
                # print(f"datas: {datas}")
                for data in batch_data:
                    # print(f"data: {data}")
                    choices, target, is_generate, correct_idx, num_choices, _ = HF_calculate_answer(ds, data, dataset_name, model, eval_type, is_generate=is_generate, dod_baselang=lang)
                    # print(f"choices: {choices}\ntarget: {target}")
                    assert len(choices) == len(target), "length choices and target must be the same!"
                    batched_prompts.extend(choices)
                    batched_continuations.extend(target)
                    batched_correct_idx.append(correct_idx)
                # print(f"batched_prompts: {batched_prompts}")
                # print(f"batched_continuations: {batched_continuations}")
                # print(f"batched_correct_idx: {batched_correct_idx}")
                
                assert num_choices, "num choices should not be None"
                # print(f"len(batched_prompts): {len(batched_prompts)}")
                # print(f"len(batched_continuations): {len(batched_continuations)}")
                # total_len = [
                #     len(
                #         model.tokenizer(p + " " + c, return_tensors="pt")
                #         .to(model.device)["input_ids"][0]
                #     )
                #     for p, c in zip(batched_prompts, batched_continuations)
                # ]
                # print(f"Max input length: {max(total_len)} | Avg: {sum(total_len) / len(total_len):.2f}")
                input_ids, attn_mask = tokenize_batch(model, batched_prompts, batched_continuations)
                if intervention:
                    # hook.intervensi_w_target_lang(model, "lape", lsn_langs, target_lang, max_new_tokens, operation_non_target, operation_target, range_layers)
                    clean_hooks(model)
                    for i in (range_layers):
                        mlp = model.model.model.layers[i].mlp
                        handlers.append(mlp.act_fn.register_forward_hook(
                            set_activation_mlp_v2(
                                replace_method=replace_method, replacer_tensor=replacer_tensor, model_name=model.model_name, name=f"{i}", lsn_langs=lsn_langs, 
                                target_lang=target_lang, operation_non_target=operation_non_target, 
                                operation_target=operation_target, attn_mask=attn_mask)))
                log_probs = calculate_logprob_batch(model, input_ids, attn_mask, batched_prompts, batched_continuations)
                log_probs = np.array(log_probs).reshape(len(batch_data), num_choices)
                predictions = log_probs.argmax(axis=1)
                result_per_lang['pred'].extend(predictions)
                result_per_lang['gold'].extend(batched_correct_idx)
                
            # if eval_type.startswith("EVAL_PPL"):
            #     eval_type, choices[correct_idx], target, is_generate
            #     perplexity = HF_calculate_answer(ds, data, dataset_name, model, text, eval_type, is_generate=is_generate)
            #     perplexity_gold = calc_perplexity_answer(eval_type, choices[correct_idx], target, model, is_generate)
            #     result_per_lang['gold'].append(perplexity)
            #     if show_df_per_lang:
            #         df_per_lang_rows = HF_make_df_per_lang(df_per_lang_rows, prompt, eval_type, gold_log_prob=perplexity)
            
            if eval_type == "EVAL_PPL":
                batched_prompts = []
                batched_continuations = []
                batched_correct_idx = []
                num_choices = None
                # print(f"datas: {datas}")
                for data in batch_data:
                    # print(f"data: {data}")
                    eval_type, choices[correct_idx], target, is_generate = HF_calculate_answer(ds, data, dataset_name, model, eval_type, is_generate=is_generate, dod_baselang=lang)
                    # print(f"choices: {choices}\ntarget: {target}")
                    assert len(choices) == len(target), "length choices and target must be the same!"
                    batched_prompts.extend(choices)
                    batched_continuations.extend(target)
                    batched_correct_idx.append(correct_idx)
                # print(f"batched_prompts: {batched_prompts}")
                # print(f"batched_continuations: {batched_continuations}")
                # print(f"batched_correct_idx: {batched_correct_idx}")
                
                assert num_choices, "num choices should not be None"
                # print(f"len(batched_prompts): {len(batched_prompts)}")
                # print(f"len(batched_continuations): {len(batched_continuations)}")
                # total_len = [
                #     len(
                #         model.tokenizer(p + " " + c, return_tensors="pt")
                #         .to(model.device)["input_ids"][0]
                #     )
                #     for p, c in zip(batched_prompts, batched_continuations)
                # ]
                # print(f"Max input length: {max(total_len)} | Avg: {sum(total_len) / len(total_len):.2f}")
                input_ids, attn_mask = tokenize_batch(model, batched_prompts, batched_continuations)
                if intervention:
                    # hook.intervensi_w_target_lang(model, "lape", lsn_langs, target_lang, max_new_tokens, operation_non_target, operation_target, range_layers)
                    clean_hooks(model)
                    for i in (range_layers):
                        mlp = model.model.model.layers[i].mlp
                        handlers.append(mlp.act_fn.register_forward_hook(
                            set_activation_mlp_v2(
                                replace_method=replace_method, replacer_tensor=replacer_tensor, model_name=model.model_name, name=f"{i}", lsn_langs=lsn_langs, 
                                target_lang=target_lang, operation_non_target=operation_non_target, 
                                operation_target=operation_target, attn_mask=attn_mask)))
                perplexity_gold = calc_perplexity_batch(eval_type, choices[correct_idx], target, model, is_generate)
                # perplexity_gold = calc_perplexity_answer(eval_type, choices[correct_idx], target, model, is_generate)
                # log_probs = calculate_logprob_batch(model, input_ids, attn_mask, batched_prompts, batched_continuations)
                # log_probs = np.array(log_probs).reshape(len(batch_data), num_choices)
                # predictions = log_probs.argmax(axis=1)
                result_per_lang['pred'].extend(predictions)
                result_per_lang['gold'].extend(batched_correct_idx)
            # if eval_type.startswith("DOD"):
            #     if eval_type == "DOD_NINT":
                    
            #         dod_dict = HF_calculate_answer(ds, data, dataset_name, model, text, eval_type, is_generate, dod_baselang=lang, dod_languages=dod_languages)
            #         assert dod_dict.keys() == non_int_dod[lang].keys(), f"the keys are notthe same! {dod_dict.keys()}, {non_int_dod[lang].keys()}"
            #         for key in non_int_dod[lang].keys():
            #             non_int_dod[lang][key].append(dod_dict[key]) 
            #     else:
            #         target_lang_dod = langs[target_lang]
            #         # print(f"target_lang_dod: {target_lang_dod}")
            #         difference = HF_calculate_answer(ds, data, dataset_name, model, text, eval_type, is_generate=is_generate, dod_baselang=lang, dod_languages=[lang, target_lang_dod])
                    
            #         result_per_lang['pred'].append(difference)
            #         # print(f"gold_difference: {gold_difference}")
            #         result_per_lang['gold'] = (gold_difference[lang][target_lang_dod])
            #         # print(f"result_per_lang['gold']: {result_per_lang['gold']}")
            # if max_samples and len(result_per_lang["gold"]) >= max_samples:
            # if max_samples and samples >= max_samples:
            #     break
            # cleanup
            for handler in handlers:
                handler.remove()
            
            clean_hooks(model)
                
        if eval_type == "EVAL_TASK":
            eval_per_lang = eval(predictions=result_per_lang['pred'], references=result_per_lang['gold'], metrics = metrics)
            eval_result[lang] = eval_per_lang
            

        if eval_type.startswith("EVAL_PPL"):
            eval_per_lang = eval_ppl(result_per_lang['gold'])
            eval_result[lang] = eval_per_lang

        # if eval_type.startswith("DOD_NINT"):
        #     eval_result[lang] = 0
            
        # if eval_type.startswith("DOD_INT"):
        #     eval_per_lang = eval_dod(preds=result_per_lang['pred'], refs=result_per_lang['gold'])
        #     eval_result[lang] = eval_per_lang 

        if show_df_per_lang:
            display(pd.DataFrame(df_per_lang_rows))
    column_name = scenario if scenario else "eval_per_lang"
    df_eval = pd.DataFrame([
        {"lang": lang, column_name: eval_result[lang]} 
        for lang in eval_result
    ])
    # print(df_eval)
    
    if eval_type == "DOD_NINT":
        return df_eval, non_int_dod
    return df_eval

class DatasetLanguage:
    def __init__(self, languages):
        """
        languages is a dict of id to language in a dataset (can be from interventor or dataset to infer)
        """
        self.id_lang_dict = languages
        self.lang_id_dict = {v:k for k,v in languages.items()}
    def idx_to_lang(self, idx):
        return self.id_lang_dict[idx]
    def lang_to_idx(self, lang):
        return self.lang_id_dict[lang]
    def get_all_idx(self):
        return [k for k in self.id_lang_dict.keys()]
    def get_all_lang(self):
        return [v for v in self.id_lang_dict.values()]
        

def intervention_matrix(
    # property dataset
    model, dataset_name, langs, max_samples, apply_template, batch_size, split, 
    # property intervensi
    replace_method, replacer_tensor, lsn, operation_non_target, operation_target, range_layers,target_langs=None, #target_langs dalam lang nya sesuai yg ada di lsn[1]
    # property evaluasi
    show_df_per_lang=False, metrics=None, is_generate=False, selected_langs=None, dataset_relations=None):
        lsn_neurons, lsn_languages = lsn
        df_int_matrix = pd.DataFrame()
        gold_difference = dict()
        if metrics == ["dod"]:
            df_int_matrix, gold_difference = HF_infer_dataset(
                model=model, dataset_name=dataset_name, dataset_relations=dataset_relations, langs=langs, max_samples=max_samples,is_generate=is_generate,
                apply_template=apply_template,batch_size=batch_size,
                intervention = False,
                split=split, show_df_per_lang=show_df_per_lang, metrics=metrics, scenario="baseline", selected_langs=selected_langs)
        else:
            df_int_matrix = HF_infer_dataset(
                model=model, dataset_name=dataset_name, dataset_relations=dataset_relations, langs=langs, max_samples=max_samples,is_generate=is_generate,
                apply_template=apply_template,batch_size=batch_size,
                intervention = False,
                split=split, show_df_per_lang=show_df_per_lang, metrics=metrics, scenario="baseline", selected_langs=selected_langs)
        
        # INTERVENTION PART
        target_langs = target_langs if target_langs!= None else lsn_languages.get_all_idx()
        for target_lang in target_langs:
            intv_df = HF_infer_dataset(
                model=model, dataset_name=dataset_name, dataset_relations=dataset_relations, langs=langs, max_samples=max_samples, is_generate=is_generate,
                apply_template=apply_template,batch_size=batch_size,
                intervention = True, replace_method=replace_method, replacer_tensor=replacer_tensor, lsn_langs = lsn_neurons, target_lang=target_lang, operation_non_target=operation_non_target, operation_target=operation_target, range_layers=range_layers,lsn_languages=lsn_languages,
                split=split, show_df_per_lang=show_df_per_lang, metrics=metrics, scenario=f"intv_{lsn_languages.idx_to_lang(target_lang)}", selected_langs=selected_langs, gold_difference=gold_difference)
            print(f"df_int_matrix: {df_int_matrix}")
            print(f"intv_df: {intv_df}")
            assert len(df_int_matrix) == len(intv_df), f"length {len(df_int_matrix)} is not the same as {len(intv_df)}, maybe the data is not parallel?"
            dfs = [df_int_matrix, intv_df]
            print(f"df_int_matrix: {df_int_matrix.columns}, intv_df: {intv_df.columns}")
            dfs = [df.set_index("lang") for df in dfs]
            df_int_matrix = pd.concat(dfs, axis=1).reset_index()
        return df_int_matrix


def alter_name(operation_target, operation_non_target, replacer_filename=None):
    if replacer_filename:
        return f"T_{replacer_filename}"
    def operation_replace(op):
        if op == "*":
            return "mult"
        elif op == "+":
            return "plus"
        elif op == "=" or op == ".":
            return "to"
        else: 
            raise ValueError("operand is wrong!")
    operation_target = operation_target.replace(operation_target[0], operation_replace(operation_target[0]))
    operation_non_target = operation_non_target.replace(operation_non_target[0], operation_replace(operation_non_target[0]))
    return f"T_{operation_target}_NT_{operation_non_target}"

# langs and selected_langs is in the dataset language code. 
# selected langs is the lang initial, langs is the lang intervener
# 'target_langs' HAS TO BE the same lang as 'langs' but langs is in dataset language code, 
# target langs is INDEX OF THAT LANGUAGE IN LSN.



parser = argparse.ArgumentParser(description="Run intervention_matrix with optional parameters.")

# Property dataset
parser.add_argument("--dataset_kaggle", type=str,  help="The dataset name in Kaggle, if not Kaggle then empty")
parser.add_argument("--ld_filename", type=str, help="The filename to process")
parser.add_argument("--lsn_filename", type=str, help="The filename to process")
parser.add_argument('--hf_token', type=str, default=None)
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--dataset_name', type=str, default=None)
parser.add_argument('--langs', nargs='+', default=None, help="langs of dataset to be intervened. for dod")
parser.add_argument('--max_samples', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--apply_template', action='store_true')
parser.add_argument('--split', type=str, default="test")

# Property intervensi
parser.add_argument("--dataset_kaggle_replacer", type=str, default=None,  help="The dataset name in Kaggle, if not Kaggle then empty")
parser.add_argument("--replacer_filename", type=str,default=None, help="The filename to process")
parser.add_argument('--replace_method', type=str, default=None)
parser.add_argument('--operation_non_target', type=str, default=None)
parser.add_argument('--operation_target', type=str, default=None)
parser.add_argument('--range_layers', nargs='+', type=int, default=None)
parser.add_argument('--target_langs', nargs='+', type=int, default=None)

# Property evaluasi
parser.add_argument('--show_df_per_lang', action='store_true')  # default is False
parser.add_argument('--metrics', nargs='+', default=None)
parser.add_argument('--selected_langs', nargs='+', default=None)

parser.add_argument("--kaggle_dataname_to_save", type=str, default=None, help="Dataset name for saving to Kaggle NO USERNAME!")
parser.add_argument("--is_update", action='store_true', help="Flag to update Kaggle dataset")
parser.add_argument("--parent_dir_to_save", type=str, default=None, help="Parent directory to save like /workspace for runpod")
#MODIF
args = parser.parse_args()
# args, unknown = parser.parse_known_args()
# from argparse import Namespace

# args = Namespace(
#     dataset_kaggle="inayarahmanisa/lsn-qwen05-flores",
#     lsn_filename="maplape.pt",
#     ld_filename="lang_dict",
#     dataset_kaggle_replacer="inayarahmanisa/activation-qwen05-flores",
#     replacer_filename="max.pt",
#     hf_token="***REMOVED***",
#     model_name="Qwen/Qwen2.5-0.5B-Instruct",
#     dataset_name="cambridgeltl/xcopa",
#     split="test",
#     replace_method="percent",
#     operation_non_target=".1",
#     operation_target="=10",
#     metrics="acc",
#     langs=None,
#     selected_langs=["id"],
#     kaggle_dataname_to_save=None,
#     is_update=False,
#     parent_dir_to_save="",
#     max_samples=None,
#     show_df_per_lang=False,
#     range_layers=None,
#     target_langs=[0],
#     apply_template=False,
#     batch_size = 6
# )
#END MODIF



from pprint import pprint
pprint(vars(args))




parent_dir = args.parent_dir_to_save if args.parent_dir_to_save else ""
login(args.hf_token)
model = InferenceModel(args.model_name)
range_layers = args.range_layers if args.range_layers else range(0, model.num_layers)
dataset_title_name = args.dataset_name.split("/")[1]

download_from_kaggle(args.dataset_kaggle, args.lsn_filename)
lape_neurons = torch.load(f"data/{args.lsn_filename}", weights_only=True)

download_from_kaggle(args.dataset_kaggle, args.ld_filename)
ld = torch.load(f"data/{args.ld_filename}", weights_only=True)

lsn_langs = {k:v for k,v in enumerate(lape_neurons)}

Languages_lape = DatasetLanguage(ld)
lsn = (lsn_langs, Languages_lape)
target_langs = args.target_langs if args.target_langs else Languages_lape.get_all_idx()


replacer_tensor = None
if args.dataset_kaggle_replacer:
    download_from_kaggle(args.dataset_kaggle_replacer, args.replacer_filename)
    # replacer_tensor = torch.load(f"data/{args.replacer_filename}", map_location='cpu')
    replacer_tensor_cpu = torch.load(f"data/{args.replacer_filename}", map_location='cpu')

    replacer_tensor = torch.empty_like(replacer_tensor, device='cuda')

    chunk_size = 800
    for i in range(0, replacer_tensor.size(0), chunk_size):
        end = min(i + chunk_size, replacer_tensor.size(0))
        replacer_tensor[i:end] = replacer_tensor[i:end].to('cuda')


matrix = intervention_matrix(
    model=model,
    dataset_name=args.dataset_name,
    langs=args.langs,
    max_samples=args.max_samples,
    apply_template=args.apply_template,
    batch_size=args.batch_size,
    split=args.split,
    replace_method=args.replace_method,
    replacer_tensor = replacer_tensor,
    lsn=lsn,
    operation_non_target=args.operation_non_target,
    operation_target=args.operation_target,
    range_layers=range_layers,
    target_langs=target_langs,
    show_df_per_lang=args.show_df_per_lang,
    metrics=args.metrics,
    selected_langs=args.selected_langs
)

path_res = f"{parent_dir}res"
os.makedirs(path_res, exist_ok=True)
matrix.to_csv(f"{path_res}/{alter_name(args.operation_target, args.operation_non_target, args.replacer_filename)}_{args.replace_method}_{args.model_name.split('/')[1]}_{dataset_title_name}_{args.metrics[0]}.csv")
if args.kaggle_dataname_to_save:
    save_to_kaggle(dataset_name=args.kaggle_dataname_to_save, data_dir=path_res, is_update=args.is_update)

