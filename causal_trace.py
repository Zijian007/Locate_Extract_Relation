import os
import re
from collections import defaultdict
import sys
import baukit
from sympy import N

sys.path.append('..')
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


def layername(model, num, kind = "hs"):
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind == "hs" else "." + kind}'
    if hasattr(model, "model"): #mistral-instruct-7b
        if kind == "embed":
            return "model.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        if kind == "att_v":
            return f'model.layers.{num}.self_attn.v_proj'
        if kind == "att_q":
            return f'model.layers.{num}.self_attn.q_proj'
        return f'model.layers.{num}{"" if kind == "hs" else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind == "hs" else "." + kind}'
    assert False, "unknown transformer structure"


# Utilities for dealing with tokens
def make_inputs(tokenizer, prompts, device):
    token_lists = [tokenizer.encode(p, add_special_tokens=True) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    substring = substring.replace(" ", "")
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


def predict_token(mt, prompts, device, return_p=False):
    inp = make_inputs(mt.tokenizer, prompts,device)
    preds, p = predict_from_input(mt.model, inp) #index and probability
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result


def predict_from_input(model, inp):
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1) #batch_size, seq_len, vocab_size fetch last token
    p, preds = torch.max(probs, dim=1) #get the max probability and its index
    return preds, p


def is_string_contains(str1, str2):
    if str1 and str2:  # 仅当 str1 和 str2 均不为空时才执行检查
        str1_lower = str1.lower()
        str2_lower = str2.lower()
        return str1_lower in str2_lower or str2_lower in str1_lower
    else:
        return False

def get_instruction_range(prompt_ids, tokenizer):
    prompt_end_id = tokenizer.convert_tokens_to_ids(".")
    # ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    prompt_ids = prompt_ids.tolist()
    prompt_end_index = prompt_ids.index(prompt_end_id)
    return (1, prompt_end_index+1) 

def get_subject_range(prompt_ids, tokenizer):
    Subject_start_id = tokenizer.convert_tokens_to_ids(".") or tokenizer.convert_tokens_to_ids(",")
    Subject_end_id = tokenizer.convert_tokens_to_ids(":")
    # ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    prompt_ids = prompt_ids.tolist()
    Subject_start_index = prompt_ids.index(Subject_start_id)
    Subject_end__index = prompt_ids.index(Subject_end_id)
    return (Subject_start_index + 1, Subject_end__index) 



def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise = 0.1,  # Level of noise to add
    trace_layers=None,  # List of traced outputs to return
):
    prng = np.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), baukit.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)
        # target_index =  generate_manually_with_for_tracing_target_index(model, mt.tokenizer, inp["input_ids"], max_length=18, target_inp=answers_t)
        # out = generate_manually_with_for_tracing_corrupted(model, mt.tokenizer, inp["input_ids"], max_length=18)
    # We report softmax probabilities for the answers_t token predictions of interest.
    # outputs_exp = out[ target_index ]
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs

def trace_important_states(model, num_layers, inp, e_range, answer_t, noise=0.1):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            print(f"Tracing {tnum}/{ntoks} ,{layer}/{num_layers}", end="\r")
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)

def trace_important_window(
    model, num_layers, inp, e_range, answer_t, kind, window = 10, noise=0.1
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            print(f"Tracing {tnum}/{ntoks} ,{layer}/{num_layers}")
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model, inp, layerlist, answer_t, tokens_to_mix = e_range, noise=noise
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)

def calculate_hidden_flow(
    mt, prompt, samples = 10, noise = 0.1, window = 1, kind = None, noise_range = None):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1), device = mt.model.device)
    print(f"inp_ids={inp['input_ids'][0]}")
    inp_tokens = mt.tokenizer.convert_ids_to_tokens(inp['input_ids'][0])
    token_dict = {}
    for i, token in enumerate(inp_tokens):
        token_dict[i] = token
    print("inp_tokens:",token_dict)
    with torch.no_grad():
        answer_t, base_score = predict_from_input(mt.model, inp)
        answer_t = answer_t[0]
        base_score  = base_score[0]
        answer_token = mt.tokenizer.convert_ids_to_tokens([answer_t])
    answer = decode_tokens(mt.tokenizer, [answer_t])
    answer = answer[0]
    print(f"Answer is {answer}, Answer_id is {answer_t}, Answer_token: {answer_token}, base_score is {base_score}")
    e_range = noise_range   
    print(f"noise_range : {noise_range }")
    low_score = trace_with_patch(mt.model, inp, [], answer_t, noise_range , noise=noise).item()
    differences = trace_important_window(
        mt.model,
        mt.num_layers,
        inp,
        noise_range ,
        answer_t,
        kind=kind,
        window=window,
        noise=noise
    )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range = noise_range,
        answer=answer,
        window=window,
        kind=kind or "",
    )

def plot_hidden_flow(
    mt,
    prompt,
    samples = 10,
    noise = 0.1,
    window = 1,
    kind = None,
    modelname = None,
    savepdf = None,
    noise_range = None
):
    # if subject is None:
    #     subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt, prompt, samples = samples, noise = noise, window = window, kind = kind, noise_range = noise_range)
    plot_trace_heatmap(result, savepdf, modelname=modelname)
    return result

def plot_all_flow(mt, prompt, subject = None, noise = 0.1, modelname = None, type = None, noise_range = None, target_word = None):
    res_dict = {}
    for kind in ["hs", "mlp", "attn",]:
        if kind == "hs":
            print(f"Plotting {kind}...")
            result = plot_hidden_flow(mt, prompt, modelname = modelname, noise=noise, kind = kind, noise_range = noise_range)
            res_dict[kind] = result
        else:
            continue
        
    return res_dict

def plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None, modelname=None):
    differences = result["scores"]
    low_score = result["low_score"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"

    with plt.rc_context(rc={"font.family": "DejaVu Sans"}):
        # fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={"hs": "Oranges", "None": "Purples", "mlp": "Greens", "attn": "Reds", "att_v": "Blues", "att_q": "gray"}[
                kind
            ],
            vmin=low_score
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        if not modelname:
            modelname = "GPT"
        if kind == "hs":
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()