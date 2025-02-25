import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import numpy as np
import torch
from datasets import load_dataset
import random
import io
import json

"""
doc https://huggingface.co/docs/datasets/loading
doc https://huggingface.co/docs/datasets/process
doc https://huggingface.co/blog/llama2#how-to-prompt-llama-2
"""


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def sample_train_loaders(name, tokenizer, nsamples=128, seed=0, seqlen=2048):
    set_seed(seed)
    if "wikitext2" in name:
        traindata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train",
        )
        traindata = "\n\n".join(traindata["text"])
    elif "c4" in name:
        traindata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        traindata = "\n\n".join(traindata["text"])
    else:
        raise NotImplementedError

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, len(traindata) - seqlen * 2 - 1)
        j = i + seqlen * 2
        # breakpoint()
        trainenc = tokenizer(traindata[i:j], return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        trainloader.append(inp)
    return trainloader


def get_redpajama_train(tokenizer, percent=10, seed=3, batch_size=128, max_length=2048):
    def tokenization(example):
        return tokenizer(example["text"], truncation=True, max_length=max_length)

    if percent != 100:
        split = f"train[:{int(850000*percent/100)}]"
    else:
        split = "train"
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split=split)

    processed_dataset = dataset.map(tokenization, batched=True, batch_size=batch_size, num_proc=os.cpu_count())
    return processed_dataset


def get_english_quote(dataset_name, tokenizer):
    data = load_dataset(dataset_name)
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    return data["train"]


def get_qat_dataset(name, tokenizer, data_percent):
    if name == "red_pajama":
        data = get_redpajama_train(tokenizer, data_percent)

    elif name == "Abirate/english_quotes":
        data = get_english_quote(name, tokenizer)
    else:
        raise NotImplementedError
    data = data.shuffle()
    return data


llama_chat_format = """<s>[INST] <<SYS>>
"Below is an instruction that describes a task. Write a response that appropriately completes the request."
<</SYS>>

{{ instruction }} [/INST] {{ response }} </s>
"""


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def get_calib_data(name, tokenizer, model_id, nsamples, seqlen=2048, seed=3, use_bos=False, args=None):
    print(f" get_ptq_calib_data {name}, nsamples={nsamples}, seqlen={seqlen}, {seed}")
    cache_file = f"{args.save_path}/{name}_{model_id.replace('/','_')}_{nsamples}_{seqlen}_{seed}_bos{use_bos}.pt"
    print(f"cache_file={cache_file}")
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if os.path.exists(cache_file):
        traindataset = torch.load(cache_file)
        return traindataset
    if name == "c4":
        traindata = load_dataset(
            "allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train"
        )
        tot_text = "\n\n".join(traindata["text"])
    elif name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tot_text = "\n\n".join(traindata["text"])
    elif name == "ptb":
        traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
        tot_text = "\n\n".join(traindata["sentence"])
    elif name == "alpaca":
        # this is for chat models
        data_path = "data/alpaca_data.json"
        list_data_dict = jload(data_path)
        traindataset = []
        selected_data_dict = random.sample(list_data_dict, nsamples)
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s = llama_chat_format.format(instruction=example["instruction"], response=example["output"])
                trainenc = tokenizer(s, return_tensors="pt")
                inp = trainenc.input_ids[:, :seqlen]
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        return traindataset
    elif name == "selfgen":
        raise NotImplementedError

    else:
        raise NotImplementedError
    print(f"tot_text={len(tot_text)}")
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        txt = tot_text[i:j]
        ind = txt.find(".")
        txt = txt[ind + 1 :].strip()
        if use_bos:
            txt = tokenizer.bos_token + txt
        trainenc = tokenizer(txt, return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    torch.save(traindataset, cache_file)
    return traindataset


def get_eval_loaders(name, tokenizer):
    if "wikitext2" in name:
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    if "ptb" in name:
        valdata = load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="validation",
        )
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
        return testenc
    if "c4" in name:
        testdata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    raise NotImplementedError


def calib_fisher_info(model, calib_loader, use_cache=True, diagonal=False, args=None):
    model_id = model.config._name_or_path
    diagonal = args.diagonal
    if diagonal:
        cache_file = f"{args.save_path}/{model_id.replace('/', '_')}_calib_fisher_info_diagonal.pt"
    else:
        cache_file = f"{args.save_path}/{model_id.replace('/', '_')}_calib_fisher_info.pt"
    if os.path.exists(cache_file) and use_cache:
        all_fisher_info = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.fisher_info = all_fisher_info.get(name, None)
                if module.fisher_info is not None:
                    module.fisher_info = module.fisher_info.to(module.weight.device)
        return
    model.train()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.fisher_info = 0

    for param in model.parameters():
        param.requires_grad = True


    # Get Fisher information
    for batch_idx, batch in enumerate(tqdm(calib_loader, desc="Calculating Fisher Information")):
        input_ids = batch["input_ids"][:, :-1].to(model.device)
        labels = batch["input_ids"][:, 1:].to(model.device)
        out = model(input_ids=input_ids, labels=labels)
        out[0].backward()

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if module.weight.grad is not None:
                    if not diagonal:
                        fisher = module.weight.grad.detach().pow(2)
                    else:
                        fisher = module.weight.grad.detach().pow(2).mean(0)
                    module.fisher_info += fisher

        model.zero_grad()

    # Average and square root
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.fisher_info is not None:
                if not diagonal:
                    module.fisher_info = module.fisher_info.div(len(calib_loader))
                else:
                    module.fisher_info = module.fisher_info.div(len(calib_loader)).sqrt()

    # Remove hooks and save fisher_info
    all_fisher_info = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            if module.fisher_info is not None:
                all_fisher_info[name] = module.fisher_info

    torch.save(all_fisher_info, cache_file)
    return cache_file



from typing import Tuple, Dict, Optional, Callable
from collections import defaultdict
def estimate_fisher_weights_causal_lm(
    model,
    dataloader,
    loss_fn: Optional[Callable] = None,
    compute_full: bool = True,
):
    hidden_dim = model.config.hidden_size
    intermediate_dim = model.config.intermediate_size
    num_hidden_layers = model.config.num_hidden_layers

    n_steps_per_epoch = len(dataloader)
    model.train()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "block_sparse_moe" in name:
            module.fisher_info = 0

    
    for inputs in tqdm(dataloader, total=n_steps_per_epoch):
        if isinstance(inputs, dict):
            # 准备输入和标签
            device = next(module.parameters()).device
            input_ids = inputs["input_ids"][:, :-1].to(device)  # 去掉最后一个token
            labels = inputs["input_ids"][:, 1:].to(device)      # 去掉第一个token
            
            # 计算损失
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
        else:
            raise ValueError("Inputs must be a dictionary containing 'input_ids'")
        
        loss.backward()

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "block_sparse_moe" in name:
                if module.weight.grad is not None:
                    fisher = module.weight.grad.detach().transpose(0, 1) ** 2
                    module.fisher_info += fisher

            
    # Collect fisher_info from the modules
    fisher_info_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "block_sparse_moe" in name:
            fisher_info_dict[name] = module.fisher_info

    # Normalize each fisher_info tensor by its maximum value
    fisher_info_normalized = {
        name: fi.detach().cpu() / fi.detach().cpu().max()
        for name, fi in fisher_info_dict.items()
    }

    model_id = model.config._name_or_path
    torch.save(fisher_info_normalized, f"outputs/{model_id.replace('/','_')}_fisher_info.pt")


@torch.no_grad()
def calib_input_distribution(model, calib_loader, method, use_cache=True):
    model_id = model.config._name_or_path
    cache_file = (
        f"outputs/{model_id.replace('/','_')}_calib_input_distribution_{method}.pt"
    )
    if os.path.exists(cache_file) and use_cache:
        all_scaling_diag_matrix = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.scaling_diag_matrix = all_scaling_diag_matrix[name].to(
                    module.weight.device
                )
        return
    model.eval()
    # set hook for every Linear layers

    def hook(module, input, output):
        if "abs_mean" in method:
            abs_mean = input[0].abs().mean(dim=-2).detach().view(-1)
            module.scaling_diag_matrix += abs_mean
        elif "abs_max" in method:
            abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
            module.scaling_diag_matrix = torch.where(
                abs_max > module.scaling_diag_matrix,
                abs_max,
                module.scaling_diag_matrix,
            )
        # abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
        # module.scaling_diag_matrix += abs_max

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.scaling_diag_matrix = 0
            module.register_forward_hook(hook)

    # get activation distribution
    for batch in tqdm(calib_loader):
        # print(batch)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        model(**batch)

    # remove and save scaling_diag_matrix
    all_scaling_diag_matrix = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            all_scaling_diag_matrix[name] = module.scaling_diag_matrix
    torch.save(all_scaling_diag_matrix, cache_file)


def get_scale_info(args):
    path = args.base_model_path
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

    model.train()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    calib_loader = get_calib_data(
        "wikitext2", tokenizer, args.base_model_path, args.num_samples, seed=0, use_bos=False, args=args
    )
    save_path = calib_fisher_info(model, calib_loader, use_cache=False, diagonal=args.diagonal, args=args)
    return save_path


def post_process_fisher_info(save_paths, args):
    path = save_paths
    
    fisher_info = torch.load(save_paths, map_location="cpu")
    
    # Create new restructured dictionary
    restructured_fisher_info = {}
    
    for key in fisher_info.keys():
        # Parse the key to extract layer index and remaining path
        parts = key.split('.')
        if 'model.layers' in key:
            layer_idx = int(parts[2])  # Get layer number
            remaining_path = '.'.join(parts[3:])  # Get remaining path
            
            # Initialize layer dict if not exists
            if layer_idx not in restructured_fisher_info:
                restructured_fisher_info[layer_idx] = {}
                
            # Store value with new key structure
            restructured_fisher_info[layer_idx][remaining_path] = fisher_info[key]
    save_path = f"{args.save_path}/fisher_{args.base_model_path.split('/')[-1]}_processed.pt"
    torch.save(restructured_fisher_info, save_path)
    import os
    os.remove(path)


def inspect_fisher_info():
    # path = "/aifs4su/lilujun/SVD-MoE-merge/outputs/fisher_SmolLlamix-8x101M.pt"
    path = "/aifs4su/lilujun/SVD-MoE-merge/outputs/fisher_Mixtral-8x7B.pt"
    fisher_info = torch.load(path, map_location="cpu")
    print(fisher_info[0]['block_sparse_moe.experts.0.w1'].shape)
    # print(fisher_info)

import re
def filter_modules_by_regex(base_module, include_patterns, include_type):
    modules = {}
    for name, module in base_module.named_modules():
        if isinstance(module, include_type) and include_patterns in name:
            modules[name] = module
    return modules

def compute_gram(model, calib_loader):
    grams = {} # gram matrices for each linear layer inputs
    xn = {} # number of examples used for computing gram

    def get_gram(name):
        def hook(module, input, output):
            x = input[0].detach() # $[b,t,h]
            x = x.view(-1, x.size(-1))
            xtx = torch.matmul(x.transpose(0,1), x) # [h,h]
            if name not in grams:
                grams[name] = xtx / x.size(0)
                xn[name] = x.size(0)
            else:
                grams[name] = (grams[name] * xn[name] + xtx) / (x.size(0) + xn[name])
                xn[name] += x.size(0)
        return hook

    linear_modules = filter_modules_by_regex(model, "block_sparse_moe.experts", nn.Linear)
    handles = []
    for name, module in linear_modules.items():
        handle = module.register_forward_hook(get_gram(name))
        handles.append(handle)

    n_step = 1000
    total = n_step if n_step > 0 else len(calib_loader)
    for step, inputs in tqdm(enumerate(calib_loader), total=total, desc='Computing gram matrix'):
        if n_step > 0 and step == n_step:
            break
        
        # Move inputs to the same device as model
        inputs = {k: v.to(next(model.parameters()).device) if hasattr(v, 'to') else v 
                 for k, v in inputs.items()}
        
        with torch.no_grad():  # 添加no_grad上下文管理器，因为只需要推理
            outputs = model(**inputs)

    for handle in handles:
        handle.remove()

    return grams


def regmean(args):
    path = args.base_model_path
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

    model.train()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    calib_loader = get_calib_data(
        "wikitext2", tokenizer, args.base_model_path, 1024, seed=0, use_bos=False, args=args
    )
    grams = compute_gram(model, calib_loader)

    torch.save(grams, f"{args.save_path}/gram_{args.base_model_path.split('/')[-1]}.pt")
    


def __main__():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="mistralai/Mixtral-8x7B-v0.1", help='Path to base model')
    parser.add_argument("--num_samples", type=int, default=1024, help='Number of samples')
    parser.add_argument("--scale_type", type=str, default="fisher", choices=["fisher", "regmean_gram"])
    parser.add_argument("--save_path", type=str, default="Model", help='Path to save scale info')
    parser.add_argument("--diagonal", type=bool, default=False, help='Whether to use diagonal fisher info')
    args = parser.parse_args()

    if args.scale_type == "fisher":
        save_path = get_scale_info(args)
        post_process_fisher_info(save_path, args)
    elif args.scale_type == "regmean_gram":
        regmean(args)

if __name__ == "__main__":
    __main__()