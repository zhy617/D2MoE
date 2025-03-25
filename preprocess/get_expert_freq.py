from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.mixtral.modeling_mixtral import *
from transformers.models.qwen2_moe.modeling_qwen2_moe import *
import torch
from datasets import load_dataset
import torch.nn.functional as F
import torch
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
from functools import partial
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="", help='Path to base model')
parser.add_argument("--save_path", type=str, default="", help='Path to save expert frequency')
parser.add_argument("--model_type", type=str, default="mixtral", choices=["mixtral", "deepseek", "phi", "qwen"])
parser.add_argument("--dataset_name", type=str, default="wikitext")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_samples", type=int, default=20000)
parser.add_argument("--batch_size", type=int, default=16)

args = parser.parse_args()

path = args.base_model_path

model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True, 
                                             torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def get_expert_frequency(model, tokenizer, args, model_type, dataset_name, split, seed, max_samples = None, batch_size = 32):
    selected_layers = list(range(len(model.model.layers)))

    device = next(model.parameters()).device
    if model_type == "mixtral":
        expert_selection_counts = {i: torch.zeros(model.model.config.num_local_experts, device=device) for i in selected_layers}
    elif model_type == "deepseek":
        expert_selection_counts = {i: torch.zeros(model.model.config.n_routed_experts, device=device) for i in selected_layers}
    elif model_type == "phi": 
        expert_selection_counts = {i: torch.zeros(model.model.config.num_local_experts, device=device) for i in selected_layers}
    elif model_type == "qwen":
        expert_selection_counts = {i: torch.zeros(model.model.config.num_experts, device=device) for i in selected_layers}

    if dataset_name == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
        text_column = 'text'
    elif 'ptb' in dataset_name:
        dataset = load_dataset('ptb_text_only', 'penn_treebank', split=split)
        text_column = 'sentence'
    elif 'c4' in dataset_name:
        dataset = load_dataset(
            'allenai/c4', 
            'en', 
            data_files={'train': ['en/c4-train.00000-of-01024.json.gz']},
            streaming=True, trust_remote_code=True  # 使用流式加载来处理大数据集
        )['train']
        text_column = 'text'
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    if max_samples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(max_samples, len(dataset))))


    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    handles = []

    def hook_for_expert_counting(module, input, output, module_name=None):
        if isinstance(module, MixtralSparseMoeBlock):
            router_logits = output[1]  # Assuming the router logits are the second output
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            _, selected_experts = torch.topk(routing_weights, k=module.top_k, dim=-1)

            for expert_idx in selected_experts.unique():
                if expert_idx.item() < expert_selection_counts[module_name].size(0):
                    expert_selection_counts[module_name][expert_idx.item()] += (selected_experts == expert_idx).sum().item()
                else:
                    logger.warning(f"Expert index {expert_idx.item()} out of range for module {module_name}")
		
        if type(module).__name__ == 'PhiMoESparseMoeBlock':
            # PhiMoE 的 forward 返回 (hidden_states, router_logits)
            hidden_states, router_logits = output
            
            # 计算路由权重并获取选中的专家
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            _, selected_experts = torch.topk(routing_weights, k=module.top_k, dim=-1)
            
            # 统计每个专家被选中的次数
            for expert_idx in range(module.num_experts):  # 遍历所有可能的专家索引
                mask = (selected_experts == expert_idx)
                count = mask.sum().item()
                expert_selection_counts[module_name][expert_idx] += count

        if isinstance(module, Qwen2MoeSparseMoeBlock):
            # PhiMoE 的 forward 返回 (hidden_states, router_logits)
            router_logits = output[1]
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            _, selected_experts = torch.topk(routing_weights, k=module.top_k, dim=-1)
            
            # 统计每个专家被选中的次数
            for expert_idx in range(module.num_experts):  # 遍历所有可能的专家索引
                mask = (selected_experts == expert_idx)
                count = mask.sum().item()
                expert_selection_counts[module_name][expert_idx] += count

        if type(module).__name__ == 'MoEGate':
            router_logits = F.linear(input[0], module.weight, None)
            if module.scoring_func == 'softmax':
                routing_weights = router_logits.softmax(dim=-1)
            _, selected_experts = torch.topk(routing_weights, k=module.top_k, dim=-1, sorted=False)

            for expert_idx in range(module.n_routed_experts):  # Iterate through all possible experts
                mask = (selected_experts == expert_idx)
                count = mask.sum().item()
                expert_selection_counts[module_name][expert_idx] += count	

		

    def create_hook(layer_idx):
        """
        Creates a partial hook function for a specific layer.
        """
        return partial(hook_for_expert_counting, module_name=layer_idx)

    # Register hooks for each expert in each selected layer
    for layer_idx in selected_layers:
        layer = model.model.layers[layer_idx]
        if hasattr(layer, 'block_sparse_moe'):
            moe_module = layer.block_sparse_moe
            handle = moe_module.register_forward_hook(create_hook(layer_idx))
            handles.append(handle)
        elif hasattr(layer, 'mlp') and model_type == "deepseek":
            moe_module = layer.mlp
            if hasattr(moe_module, 'gate'):
                gate_module = moe_module.gate
                handle = gate_module.register_forward_hook(create_hook(layer_idx))
                handles.append(handle)
        elif hasattr(layer, 'mlp') and model_type == "qwen":
            moe_module = layer.mlp
            handle = moe_module.register_forward_hook(create_hook(layer_idx))
            handles.append(handle)


    # Iterate through the dataloader and perform forward passes to collect counts
    for batch in tqdm(dataloader, desc="Collecting expert activation counts"):
        inputs = tokenizer(batch[text_column], truncation=True, padding=True, max_length=2048, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)

    # Remove all hooks after collection
    for handle in handles:
        handle.remove()

    # Save the counts to a JSON file
    counts_dict = {layer: counts.tolist() for layer, counts in expert_selection_counts.items()}

    with open(f"{args.save_path}/{args.model_type}_{args.dataset_name}_{args.max_samples}_expert_frequencies.json", "w") as f:
        json.dump(counts_dict, f, indent=4)
	

get_expert_frequency(model, tokenizer, args, args.model_type, args.dataset_name, args.split, args.seed, args.max_samples, args.batch_size)
