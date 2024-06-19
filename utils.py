from ast import List
from matplotlib.pylab import f
from regex import F
from transformers import AutoModelForCausalLM, AutoTokenizer
import baukit.nethook as nethook
import re
import torch
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
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map = self.device, torch_dtype=torch_dtype)
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
