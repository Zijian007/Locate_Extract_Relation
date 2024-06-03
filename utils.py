from ast import List
from matplotlib.pylab import f
from regex import F
from transformers import AutoModelForCausalLM, AutoTokenizer
import nethook
import re
import torch
import transformers
import torch.nn as nn
import functools
import matplotlib.pyplot as plt
class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
        self,
        model_name=None,
        device = None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage = False,
        torch_dtype = torch.bfloat16,
    ):
        self.device = device
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model is None:
            assert model_name is not None
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map = self.device)
            nethook.set_requires_grad(False, model)
            model.eval()
            model = model.to(self.device)
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox|model)\.(h|layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )
    
def get_keys_from_value(dictionary, value):
    keys = []
    for key, val in dictionary.items():
        if val == value:
            keys.append(key)
    return keys

def make_inputs(tokenizer, prompts, device="cuda"):
  """Prepare inputs to the model."""
  token_lists = [tokenizer.encode(p) for p in prompts]
  maxlen = max(len(t) for t in token_lists)
  if "[PAD]" in tokenizer.all_special_tokens:
    pad_id = tokenizer.all_special_ids[
        tokenizer.all_special_tokens.index("[PAD]")
        ]
  else:
    pad_id = 0
  input_ids = [
      [pad_id] * (maxlen - len(t)) + t for t in token_lists]
  attention_mask = [
      [0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists
      ]
  return dict(
      input_ids=torch.tensor(input_ids).to(device),
      attention_mask=torch.tensor(attention_mask).to(device),
      )


def decode_tokens(tokenizer, token_array):
  if hasattr(token_array, "shape") and len(token_array.shape) > 1:
    return [decode_tokens(tokenizer, row) for row in token_array]
  return [tokenizer.decode([t]) for t in token_array]





def predict_from_input(model, inp):
  out = model(**inp)["logits"]
  probs = torch.softmax(out[:, -1], dim=1)
  p, preds = torch.max(probs, dim=1)
  return preds, p


def set_requires_grad(requires_grad, *models):
  for model in models:
    if isinstance(model, torch.nn.Module):
      for param in model.parameters():
        param.requires_grad = requires_grad
    elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
      model.requires_grad = requires_grad
    else:
      assert False, "unknown type %r" % type(model)

def generate_mask_dynamically(num_layers, query_index, key_unvisiable_index_range):
    sever_config = {}
    key_unvisiable_index_list = [i for i in range(key_unvisiable_index_range[0], key_unvisiable_index_range[1])]
    for layer in range(num_layers):
      result = [(i, j) for i in range(query_index, key_unvisiable_index_range[1]-1 , -1) for j in key_unvisiable_index_list]
      sever_config[layer] = result
    return sever_config

def set_block_attn_hooks(model, from_to_index_per_layer, opposite=False):
    """
    Only works on GPT2
    """
    def wrap_attn_forward(forward_fn, model_, from_to_index_, opposite_):
        @functools.wraps(forward_fn)
        def wrapper_fn(*args, **kwargs):
            new_args = []
            new_kwargs = {}
            for arg in args:
                new_args.append(arg)
            for (k, v) in kwargs.items():
                new_kwargs[k] = v

            hs = kwargs["hidden_states"]
            num_tokens = list(hs[0].size())[0]
            num_heads = model_.config.num_attention_heads
            if opposite_:
                attn_mask = torch.tril(torch.zeros((num_tokens, num_tokens), dtype=torch.uint8))
                for s, t in from_to_index_:
                    attn_mask[s, t] = 1
            else:
                attn_mask = torch.tril(torch.ones((num_tokens, num_tokens), dtype=torch.uint8))
                for s, t in from_to_index_:
                    attn_mask[s, t] = 0
            attn_mask = attn_mask.repeat(1, 1, 1, 1)
            attn_mask = attn_mask.to(dtype=model_.dtype)  # fp16 compatibility
            attn_mask = (1.0 - attn_mask) * torch.finfo(model_.dtype).min
            attn_mask = attn_mask.to(hs.device)

            new_kwargs["attention_mask"] = attn_mask
            return forward_fn(*new_args, **new_kwargs)
        return wrapper_fn
    hooks = []
    for i in from_to_index_per_layer.keys():
        hook = model.model.layers[i].self_attn.forward
        model.model.layers[i].self_attn.forward = wrap_attn_forward(model.model.layers[i].self_attn.forward,
                                                                model, from_to_index_per_layer[i], opposite)
        hooks.append((i, hook))
    return hooks

def remove_wrapper(model, hooks):
    for i, hook in hooks:
        model.model.layers[i].self_attn.forward = hook

def hook_att_sever(
    model,
    tokenizer,
    inp,
    from_to_index_per_layer,  #from_to_index_per_layer:{1: [(10, 0)]} {layer: [(Q, K)]}
    answers_t
):
    with torch.no_grad():
        # set hooks
        block_attn_hooks = set_block_attn_hooks(model, from_to_index_per_layer)
        # get prediction
        outputs_exp = model(**inp)
        # out = generate_manually_with_probs_with_mask(model, tokenizer, inp, max_length=10, target_inp=None)
        # remove hooks
        remove_wrapper(model, block_attn_hooks)
    # probs = torch.softmax(outputs_exp.logits[0, -1, :], dim=0)[answers_t]
    # token_pred = outputs_exp.logits[0, -1, :].argmax().item()
    # token = tokenizer.decode([token_pred])
    return outputs_exp


def generate_manually_with_probs_with_mask(model, tokenizer, inp, max_length, target_inp):
    #手动生成文本
    current_length = 0
    input_ids = inp["input_ids"]
    input_length = len(input_ids[0])
    print(f"Input_length:{input_length}")
    print("----------------------------")
    attention_mask = inp["attention_mask"]
    generated_ids = input_ids
    tokens = []
    target_probs = []
    for _ in range(max_length):
      print(f"step:{_}")
      # print(f"attention_mask:{attention_mask}")
      current_length = len(generated_ids[0])
      print(f"current length:{current_length}")
      if _>=1:
        from_to_index_per_layer = generate_mask_dynamically(num_layers = 32, query_index = current_length-1, key_unvisiable_index_range = (1, input_length-1))
        # print(f"from_to_index_per_layer:{from_to_index_per_layer}")
        block_attn_hooks = set_block_attn_hooks(model, from_to_index_per_layer)
        outputs = model(generated_ids, attention_mask, output_hidden_states=True, output_attentions=True)
        remove_wrapper(model, block_attn_hooks)
      else:
        # print("No att sever")
        # hooks = []
        # for i in range(31):
        #     # print(i)
        #     hook = model.model.layers[i].self_attn.forward
        #     hooks.append((i, hook))
        # remove_wrapper(model, hooks)
        outputs = model(generated_ids, attention_mask, output_hidden_states=True, output_attentions=True)
      next_token_logits = outputs.logits[:, -1, :]
      probs = nn.functional.softmax(next_token_logits, dim=-1)
      att_weights = outputs.attentions
      # print(f"att_weights:{att_weights[3][0][12]}")
      # greedy decoding
      next_token_id = next_token_logits.argmax(1)
      next_token_id = next_token_id.unsqueeze(-1)
      next_token = tokenizer.batch_decode(next_token_id)
      attention_mask = torch.cat((attention_mask, torch.ones(1,1).to(attention_mask.device)), dim=1)
      if _ == 0:
        ids = next_token_id
      else:
        ids = torch.cat((ids, next_token_id), dim=1)
      generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

    generated_text = tokenizer.batch_decode(generated_ids)
    new_text = tokenizer.batch_decode(ids)
    return generated_text , new_text, target_probs


def generate_manually_with_probs(model, tokenizer, inp, max_length, target_inp):
    #手动生成文本
    input_ids = inp["input_ids"]
    generated_ids = input_ids
    if target_inp is not None:
      if isinstance(target_inp, int):
        target_token_ids = target_inp
      else:
        target_token_ids = target_inp["input_ids"]
    # print(f"subtoken_ids:{subtoken_ids}")
    tokens = []
    target_probs = []
    for _ in range(max_length+1):
      # get logits and hidden states
      outputs = model(generated_ids, output_hidden_states=True)
      next_token_logits = outputs.logits[:, -1, :]
      probs = nn.functional.softmax(next_token_logits, dim=-1)
      if target_inp is not None:
        if isinstance(target_inp, int):
          target_prob = probs[0, target_token_ids].item()
        else:
          target_prob = probs[0, target_token_ids[0][-1]].item()
      # greedy decoding
      next_token_id = next_token_logits.argmax(1)
      next_token_id = next_token_id.unsqueeze(-1)
      next_token = tokenizer.batch_decode(next_token_id)
      if _ == 0:
        ids = next_token_id
      else:
        ids = torch.cat((ids, next_token_id), dim=1)
      # tokens.append(next_token)
      if target_inp is not None:
        target_probs.append(target_prob)
      # if next_token in ["<0x0A>"]:
      #     continue
      # if next_token == tokenizer.eos_token:
      #     break
      generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
      if target_inp is not None and next_token_id.item() == target_token_ids[0][-1]:
          print(f"Meet target token {next_token}, the probability is {target_prob}")
          break
    # print("Predicted tokens:",tokens)
    generated_text = tokenizer.batch_decode(generated_ids)
    new_text = tokenizer.batch_decode(ids)
    # print(f"ids:{ids}")
    return generated_text , new_text, target_probs

def generate_manually_with_for_tracing_target_index(model, tokenizer, inp, max_length, target_inp):
    #手动生成文本
    input_ids = inp[0]
    generated_ids = input_ids.unsqueeze(0)
    if target_inp is not None:
        target_token_ids = target_inp
    # print(f"subtoken_ids:{subtoken_ids}")
    ids = []
    tokens = []
    target_probs = []
    for _ in range(max_length):
        # get logits and hidden states
        outputs = model(generated_ids, output_hidden_states=True)
        next_token_logits = outputs.logits[:, -1, :]
        probs = nn.functional.softmax(next_token_logits, dim=-1)
        if target_inp is not None:
            target_prob = probs[0, target_token_ids].item()
        # greedy decoding
        next_token_id = next_token_logits.argmax(1).unsqueeze(-1)
        next_token = tokenizer.convert_ids_to_tokens(next_token_id.item())
        ids.append(next_token_id.item())
        tokens.append(next_token)
        if target_inp is not None:
          target_probs.append(target_prob)
        generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
        if target_inp is not None and next_token_id.item() == target_token_ids:
            target_index = _
            print(f"Meet target token {next_token} at step {target_index}, the probability is {target_prob}", end=" \r")
            
            break
    # print(tokens)
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    new_text = tokenizer.decode(ids)
    # return generated_text , new_text, torch.tensor(target_probs[-1])
    return target_index 

def generate_manually_with_for_tracing_corrupted(model, tokenizer, inp, max_length):
    #手动生成文本
    input_ids = inp
    generated_ids = input_ids
    ids = []
    tokens = []
    logits = []
    outputs = []
    for _ in range(max_length):
        # get logits and hidden states
        output = model(generated_ids, output_hidden_states=True)
        outputs.append(output)
        next_token_logits = output.logits[:, -1, :]
        logits.append(next_token_logits)
        probs = nn.functional.softmax(next_token_logits, dim=-1)
        # greedy decoding
        next_token_id = next_token_logits.argmax(1).unsqueeze(-1)
        # next_token = tokenizer.convert_ids_to_tokens(next_token_id.item())
        # ids.append(next_token_id.item())
        # tokens.append(next_token)
        generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

    # print(tokens)
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    new_text = tokenizer.decode(ids)
    # return generated_text , new_text, torch.tensor(target_probs[-1])
    return outputs


def plot_scores(score_no_rel, score_no_sub, title):
    x1 = torch.arange(len(score_no_rel))
    x2 = torch.arange(len(score_no_sub))
    plt.figure(figsize=(10, 4)) 
    plt.plot(x1, score_no_rel, color='red', linewidth=1.0, linestyle='--', label='ME of Relation')
    plt.plot(x2, score_no_sub, color='blue', linewidth=1.0, linestyle='--', label='ME of Subject')
    new_xticks = torch.arange(len(score_no_rel))
    plt.xticks(new_xticks)
    plt.title(title)
    plt.xlabel("layer")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

  
def find_token_range(tokenizer, prompt, substring):
    toks = tokenizer.tokenize(prompt, add_special_tokens = True)
    # print(toks)
    toks = [item.replace('▁', ' ') if item.startswith('▁') else item for item in toks[:]]
    print("Tokens:",toks)
    whole_string = "".join(toks[1:])
    # print(whole_string )
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks[1:]):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return [tok_start+1, tok_end+1]
