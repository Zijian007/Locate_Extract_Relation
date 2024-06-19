import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch
import baukit
import torch.nn as nn
import unicodedata
import re
import json 

def get_reading_layers(config, start=0, end=32):
    num_layers = config.num_hidden_layers
    assert start < end and end <= num_layers, print("Invalid layer range, total layers: ", num_layers)
    layer_start = start
    layer_end = end
    layer_range_reading = range(layer_start, layer_end)
    return layer_range_reading

def extract_task_vectors_from_last(model, tokenizer, input_text, layers,  rep_token = -1):
    hidden_states_layers = {}
    encoded_inputs = tokenizer(input_text, return_tensors='pt')
    input_ids = encoded_inputs['input_ids'].to(model.device)
    model_outputs = model(input_ids, use_cache=True, output_hidden_states=True,output_attentions=True)['hidden_states']
    # print(type(model_outputs))
    for layer in layers:
        hidden_states = model_outputs[layer]#(batch_size, sequence_length, hidden_size).
        hidden_states =  hidden_states[:, rep_token, :]#(batch_size,hidden_size)
        # hidden_states_layers[layer] = hidden_states.cpu().to(dtype=torch.float32).detach().numpy()
        hidden_states_layers[layer] = hidden_states.detach()
    return hidden_states_layers

def get_control_layers(config, kind ="hs", start=0, end=32):
    num_layers = config.num_hidden_layers
    assert start < end and end <= num_layers, print("Invalid layer range, total layers: ", num_layers)
    layer_names = []
    layer_range_control = range(start, end)
    for layer in layer_range_control:
        if kind == "att":
            layer_name = f'model.layers.{layer}.self_attn.v_proj'
        elif kind == "mlp":
            layer_name = f'model.layers.{layer}.mlp'
        elif kind == "hs":
            layer_name = f'model.layers.{layer}'
        layer_names.append(layer_name)
    assert len(layer_names) == len(layer_range_control), print("length of layers_names_for_control and layer_range_control should be the same")
    return layer_range_control, layer_names

 
def create_replace_function(task_vectors, coff_task, layers_names_controlable, layer_range_names_for_control_task):
    input_token_len = None 
    def replace(module,input,output,_count=[0]):
        nonlocal input_token_len  
        _count[0] += 1
        if _count[0] == 1:
            input_token_len = input[0].shape[1]
        layer_idx = layers_names_controlable.index(module)
        if module in layer_range_names_for_control_task:
            # print(f"{module} for Task editing, The coff is {coff_task}")
            output = list(output)
            output[0][:,input_token_len -1 ,:] = max((1 - coff_task),0)* output[0][:,input_token_len -1,:] + coff_task * task_vectors[layer_idx]
            output = tuple(output)
            return output
        else:
            # print(f"{module} for No editing")
            output = output
            return output
    return replace

def plot_scores(score_no_instruct, score_no_input, title):
    x1 = torch.arange(len(score_no_instruct))
    x2 = torch.arange(len(score_no_input))
    plt.figure(figsize=(10, 4)) 
    plt.plot(x1, score_no_instruct, color='red', linewidth=1.0, linestyle='--', label='ME of Subject')
    plt.plot(x2, score_no_input, color='blue', linewidth=1.0, linestyle='--', label='ME of Subject')
    new_xticks = torch.arange(len(score_no_instruct))
    plt.xticks(new_xticks)
    plt.title(title)
    plt.xlabel("layer")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    
class SubObjDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        subject = item["subject"]
        object = item["object"]
        return {"subject": subject, "object": object}
    
def is_string_contains(str1, str2):
    str1_normalized = unicodedata.normalize('NFC', str1).encode('ASCII', 'ignore').decode('utf-8')
    str2_normalized = unicodedata.normalize('NFC', str2).encode('ASCII', 'ignore').decode('utf-8')

    str1_normalized = re.sub(r'[^\w\s]', '', str1_normalized)
    str2_normalized = re.sub(r'[^\w\s]', '', str2_normalized)

    str1_lower = str1_normalized.lower()
    str2_lower = str2_normalized.lower()

    return str1_lower in str2_lower or str2_lower in str1_lower

def create_dataloader(json_path,batch_size=1):
    parsed_data = json.load(open(json_path, 'r'))
    parsed_data = parsed_data["samples"]
    dataset = SubObjDataset(parsed_data)
    len_data = len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for i in range(len_data):
        data = dataset[i]
        subject = data["subject"]
        object = data["object"]
        print(f"subject:{subject},object:{object}")
        break
    return dataloader, len_data

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

def generate_manually_with_probs(model, tokenizer, inp, max_length):
    #手动生成文本
    input_ids = inp["input_ids"]
    generated_ids = input_ids
    # print(f"subtoken_ids:{subtoken_ids}")
    tokens = []
    for _ in range(max_length+1):
      # get logits and hidden states
      outputs = model(generated_ids, output_hidden_states=True)
      next_token_logits = outputs.logits[:, -1, :]
      probs = nn.functional.softmax(next_token_logits, dim=-1)
      # greedy decoding
      next_token_id = next_token_logits.argmax(1)
      next_token_id = next_token_id.unsqueeze(-1)
      next_token = tokenizer.batch_decode(next_token_id)
      if _ == 0:
        ids = next_token_id
      else:
        ids = torch.cat((ids, next_token_id), dim=1)
      tokens.append(next_token)
      generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
    # print("Predicted tokens:",tokens)
    generated_text = tokenizer.batch_decode(generated_ids)
    new_text = tokenizer.batch_decode(ids)
    # print(f"ids:{ids}")
    return generated_text , new_text, probs

def zero_shot(mt, json_path,  Ref_text, layer_range = (0, 18), coff = 1, template = ""):
    _,layer_names_controlable = get_control_layers(mt.model.config, kind="hs", start=0, end=32)
    task_vectors = extract_task_vectors_from_last(mt.model, mt.tokenizer, Ref_text, range(layer_range[0], layer_range[1]),  rep_token = -1)
    layer_rang_id_for_control_task, layer_range_names_for_control_task = get_control_layers(mt.model.config, kind="hs", start = layer_range[0], end = layer_range[1]) 
    parsed_data = json.load(open(json_path, 'r'))
    Template= parsed_data["Prompt"]
    dataloader, len_data = create_dataloader(json_path, batch_size=1)
    acc_zs, acc_full, acc_zs_num, acc_full_num, Bad_cases, Good_cases = 0, 0, 0, 0, [], []
    for i,batch in enumerate(dataloader):
        Bad_case, Good_case = [], []
        Sub, Obj = batch["subject"][0], batch["object"][0]
        Input = template.format(Sub)
        print(f"Input:{Input}")
        Input_token_id = make_inputs(mt.tokenizer, [Input], mt.device)
        Template_inp = make_inputs(mt.tokenizer, [Template.format(Sub)], device = mt.device)
        edited_out, new_out_full ,_ = generate_manually_with_probs(mt.model, mt.tokenizer, Template_inp , max_length = 5)
        with baukit.TraceDict(
            mt.model, layers = layer_names_controlable, 
            edit_output = create_replace_function(task_vectors = task_vectors, coff_task = coff, layers_names_controlable = layer_names_controlable,                                                                             
                                                    layer_range_names_for_control_task = layer_range_names_for_control_task), 
                                                    retain_input = True, retain_output=True
        ) as ret:
            # print("--------------------------------------------------------------")
            edited_out, new_out, _ = generate_manually_with_probs(mt.model, mt.tokenizer, Input_token_id, max_length = 5)
            new_out, new_out_full = new_out[0], new_out_full[0]
            print(f"progress:{i}/{len_data}\nSub: {Sub}\nResult_ZS:{is_string_contains(new_out, Obj)}, ZS Obj: {new_out}, True Obj: {Obj}\nResult_Full:{is_string_contains(new_out_full, Obj)}, Full Obj: {new_out_full}, True Obj: {Obj}")
            acc_zs_num += is_string_contains(new_out, Obj)
            acc_full_num += is_string_contains(new_out_full, Obj) 
            acc_zs = acc_zs_num/len_data
            acc_full = acc_full_num/len_data
            print(f"acc_zs:{acc_zs}, acc_full:{acc_full}")
            print("--------------------------------------------------------------")
            if is_string_contains(new_out, Obj) == False:
                Bad_case = {"Subject": Sub, "Predict_Obj": new_out, "True_Obj": Obj}
                Bad_cases.append(Bad_case)
            if is_string_contains(new_out_full, Obj) == True:
                Good_case = {"Subject": Sub, "Predict_Obj": new_out_full, "True_Obj": Obj}
                Good_cases.append(Good_case)         
    # print(Bad_cases)
    return acc_zs, acc_full, Good_cases

Relations = ["the size of", "the weight of", "the height of", "the length of", "the width of", "the temperature of", "the speed of", 
"the duration of", "the age of", "the volume of", "the density of", "the brightness of", "the intensity of", "the depth of", 
"the pressure of", "the sound of", "the texture of", "the smell of", "the taste of", "the shape of", "the color of", 
"the pattern of", "the frequency of", "the resolution of", "the power of", "the energy of", "the efficiency of", "the accuracy of",
"the precision of", "the complexity of", "the simplicity of", "the clarity of", "the transparency of", "the opacity of", 
"the fragility of", "the flexibility of", "the rigidity of", "the brittleness of", "the hardness of", "the softness of",
"the future of","the acidity of" "the smoothness of", "the roughness of", "the elasticity of", "the conductivity of",
"the resistance of", "the magnetism of", "the gravity of", "the buoyancy of", "the transparency of", "the reflectivity of",
"the absorbency of", "the reactivity of", "the stability of", "the volatility of", "the solubility of", "the viscosity of",
"the alkalinity of", "the composition of", "the structure of", "the organization of", "the arrangement of","the formation of", 
"the development of", "the growth of", "the evolution of", "the behavior of", "the function of","the purpose of", "the role of", 
"the significance of", "the impact of", "the influence of", "the effect of", "the result of","the outcome of", "the consequence of", 
"the importance of", "the relevance of", "the connection of", "the relationship of","the interaction of", "the correlation of", 
"the similarity of", "the difference of", "the contrast of", "the comparison of","the similarity of", "the variation of", 
"the change of", "the improvement of", "the innovation of", "the advancement of","the discovery of", "the invention of", 
"the application of", "the utilization of", "the adaptation of", "the transformation of"]

def edit(mt, data, Ref_text, new_relation, layer_range = (0, 18), coff_task = 1):
    _,layer_names_controlable = get_control_layers(mt.model.config, kind="hs", start=0, end=32)
    task_vectors = extract_task_vectors_from_last(mt.model, mt.tokenizer, Ref_text, range(layer_range[0], layer_range[1]),  rep_token = -1)
    layer_rang_id_for_control_task, layer_range_names_for_control_task = get_control_layers(mt.model, mt.model.config, kind="hs", start = layer_range[0], end = layer_range[1]) 
    len_data = len(data)
    acc, acc_modify, acc_num, acc_num_modify = 0, 0, 0, 0
    for i in range(len_data):
        Sub = data[i]['Subject']
        Obj = data[i]['True_Obj']
        relation = random.choice(Relations)
        Input = "What is {} {}?".format(relation, Sub)
        Input_modify = Input + f" Actually, I am asking its {new_relation}"
        print(f"Input: {Input}")
        print(f"Input_modify: {Input_modify}")
        Input_token_id = make_inputs(mt.tokenizer, [Input], mt.device)
        Input_modify_token_id = make_inputs(mt.tokenizer, [Input_modify], mt.device)
        edited_out, new_out_modify, _ = generate_manually_with_probs(mt.model, mt.tokenizer,  Input_modify_token_id, max_length = 5)

        with baukit.TraceDict(
            mt.model, layers = layer_names_controlable, 
            edit_output = create_replace_function(task_vectors = task_vectors, coff_task = coff_task, layers_names_controlable = layer_names_controlable,                                                                             
                                                    layer_range_names_for_control_task = layer_range_names_for_control_task), 
                                                    retain_input = True, retain_output=True
        ) as ret:
            # print("--------------------------------------------------------------")
            edited_out, new_out, _ = generate_manually_with_probs(mt.model, mt.tokenizer, Input_token_id, max_length = 5)
        new_out, new_out_modify = new_out[0], new_out_modify[0]
        acc_num += is_string_contains(new_out, Obj)
        acc_num_modify += is_string_contains(new_out_modify, Obj)
        acc = acc_num/len_data
        acc_modify = acc_num_modify/len_data
        print(f"progress:{i}/{len_data}\nSub: {Sub}\nResult:{is_string_contains(new_out, Obj)}\n Edit_Obj: {new_out}, Modify_obj: {new_out_modify}, True_Obj: {Obj}")
        print(f"acc:{acc}, acc_modify:{acc_modify}")
        print("--------------------------------------------------------------")
    return acc

def plot_scores_sss(score_no_instruct, score_no_input, title, variance_scale=1):
    mean_no_instruct = np.mean(score_no_instruct, axis=0)
    mean_no_input = np.mean(score_no_input, axis=0)
    std_no_instruct = np.std(score_no_instruct, axis=0) * variance_scale
    std_no_input = np.std(score_no_input, axis=0) * variance_scale
    
    x = np.arange(len(mean_no_instruct))  
    
    plt.figure(figsize=(10, 4))
    ax = plt.gca() 
    plt.plot(x, mean_no_instruct, color='red', linewidth=1.0, linestyle='--', label='Mean - No Instruct')
    plt.plot(x, mean_no_input, color='green', linewidth=1.0, linestyle='--', label='Mean - No Input')
    plt.fill_between(x, mean_no_instruct - std_no_instruct, mean_no_instruct + std_no_instruct, color='lightcoral', alpha=0.2, edgecolor='none')
    plt.fill_between(x, mean_no_input - std_no_input, mean_no_input + std_no_input, color='lightgreen', alpha=0.2, edgecolor='none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel("Mediating Effect")
    # plt.legend()
    plt.show()
