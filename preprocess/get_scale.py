from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.mixtral.modeling_mixtral import *
from transformers.models.qwen2_moe.modeling_qwen2_moe import *
import torch
from datasets import load_dataset
import torch.nn.functional as F
import torch
from tqdm import tqdm
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="", help='Path to base model')
parser.add_argument("--save_path", type=str, default="", help='Path to save SVD scale')
parser.add_argument("--model_type", type=str, default="mixtral", choices=["mixtral", "deepseek", "phi", "qwen"])
parser.add_argument("--dataset_name", type=str, default="wikitext", choices=["wikitext", "ptb", "c4"])
parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_samples", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--selected_layers", type=str, default=None, help='Comma separated list of layer indices to select')
parser.add_argument("--seqlen", type=int, default=2048)

args = parser.parse_args()

path = args.base_model_path

model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True, 
                                             torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def get_free_gpu():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return 'cpu'
    free_memory = [torch.cuda.mem_get_info(i)[0] for i in range(num_gpus)]
    most_free_gpu_index = int(torch.argmax(torch.tensor(free_memory)))
    return f'cuda:{most_free_gpu_index}'

def process_scaling_matrix(raw_matrix, name, module_device):
    """Helper function to process scaling matrix"""
    if isinstance(raw_matrix, (float, int)):
        return raw_matrix

    matrix = raw_matrix.clone()

    # Ensure the matrix is symmetric
    matrix = (matrix + matrix.T) / 2

    # 进行特征值分解
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    
    # 检查最小特征值
    min_eigenval = eigenvalues.min().item()
    if min_eigenval <= 0:
        # 如果特征值接近0（比如小于1e-5），将其调整为一个小的正数
        eigenvalues = torch.clamp(eigenvalues, min=1e-5)
        # 用调整后的特征值重构矩阵
        matrix = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T

    # 进行 Cholesky 分解
    # cholesky_matrix = torch.linalg.cholesky(matrix)
    # 添加一个非常小的正数到对角线，保证矩阵是正定的
    eps = 1e-6 
    cholesky_matrix = torch.linalg.cholesky(matrix + eps * torch.eye(matrix.shape[0], device=matrix.device))
    return cholesky_matrix


def find_layers(module, layers=[nn.Conv2d, nn.Linear, MixtralSparseMoeBlock], name='', process_moe_block=False):
    res = {}

    if isinstance(module, MixtralSparseMoeBlock) or type(module).__name__ == 'MoEGate' or type(module).__name__ == 'PhiMoESparseMoeBlock':
        if process_moe_block:
            res[name] = module
            for name1, child in module.named_children():
                res.update(find_layers(
                    child, layers=layers, name=name + '.' + name1 if name != '' else name1, process_moe_block=process_moe_block
                ))
        else:
            for name1, child in module.named_children():
                res.update(find_layers(
                    child, layers=layers, name=name + '.' + name1 if name != '' else name1, process_moe_block=False
                ))
        return res 
    elif type(module) in layers or 'gate' in name:
        res[name] = module
    else:
        for name1, child in module.named_children():
            res.update(find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1, process_moe_block=process_moe_block
            ))
    return res

def find_linear_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res



@torch.no_grad()
def get_svd_scale(model, tokenizer, model_name, dataset_name='wikitext', split='train', 
                    seed=42, seqlen=2048, batch_size=1, max_samples=None, selected_layers=None):
    layers = model.model.layers
    layers[0] = layers[0]
    if dataset_name == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
        text_column = 'text'
        tot_text = "\n\n".join(dataset[text_column])

    elif 'ptb' in dataset_name:
        dataset = load_dataset('ptb_text_only', 'penn_treebank', split=split)
        text_column = 'sentence'
        tot_text = "\n\n".join(dataset[text_column])

    elif 'c4' in dataset_name:
        dataset = load_dataset(
            'allenai/c4', 
            'en', 
            data_files={'train': ['en/c4-train.00000-of-01024.json.gz']},
            streaming=True, trust_remote_code=True  # 使用流式加载来处理大数据集
        )['train']
        text_column = 'text'
        tot_text = "\n\n".join(dataset[text_column])

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    calib_loader = []
    for s in range(max_samples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            s = s - 1
            continue
        if s % batch_size == 0:
            if s != 0:
                attention_mask = torch.ones_like(inp)
                calib_loader.append({"input_ids": inp, "attention_mask": attention_mask})
            inp = trainenc.input_ids[:, :seqlen]
        else:
            inp = torch.cat((inp, trainenc.input_ids[:, :seqlen]), dim=0)

    if selected_layers is None:
        selected_layers = list(range(len(layers)))
    else:
        selected_layers = selected_layers
    

    dtype = next(iter(model.parameters())).dtype
    device = get_free_gpu()

    inps = torch.zeros(
        (len(calib_loader), seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.detach()
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask'].detach()), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids'].detach()), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])

    for batch in calib_loader:
        try:
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass

    layers[0] = layers[0].module

    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids'].detach()
    profiling_mat = {}
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        process_subset = {}
        layer = layers[i]
        # subset = find_layers(module = layer, layers=[nn.Linear, MixtralSparseMoeBlock], process_moe_block=True) 
        if i in selected_layers:
            subset = find_linear_layers(module = layer, layers=[nn.Linear])

            for name, module in subset.items():
                if 'experts' in name and "shared" not in name:
                    process_subset[name] = module

        def hook(module, input, output):
            module_device = next(module.parameters()).device
            inp = input[0].detach().float().to(module_device)
            if inp.dim() == 2:  # for opt
                inp = inp.unsqueeze(0)
            inp = inp.view(-1, inp.size(-1))
            adds = torch.matmul(inp.transpose(0, 1), inp)

            # min_eigenval = torch.linalg.eigvalsh(adds)[0].item()
            # if min_eigenval < 0:
            #     adjustment = (-min_eigenval + 1e-6)
            #     adds += adjustment * torch.eye(adds.shape[0], device=module_device, dtype=torch.bfloat16)

            module.scaling_diag_matrix += adds
            del inp, adds, output
            torch.cuda.empty_cache()
        
        handles = []
        
        for name in process_subset:
            process_subset[name].scaling_diag_matrix = 0
            handles.append(process_subset[name].register_forward_hook(hook))

        for j in range(inps.shape[0]):
            if attention_masks is not None:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(device))[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0))[0]
        for h in handles:
            h.remove()
        torch.cuda.empty_cache()
        for name in process_subset:
            device = get_free_gpu()
            raw_scaling_diag_matrix = process_subset[name].scaling_diag_matrix.double().to(device)
            scaling_diag_matrix = process_scaling_matrix(raw_scaling_diag_matrix, name, device)
            layer_profile[name] = scaling_diag_matrix.to(torch.bfloat16)
            scaling_diag_matrix = raw_scaling_diag_matrix = process_subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, process_subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        layers[i] = layer
        profiling_mat[i] = layer_profile
        inps = outs
        torch.cuda.empty_cache()
    return profiling_mat

if args.selected_layers is not None:
    selected_layers_list = [int(x) for x in args.selected_layers.split(',')]
else:
    selected_layers_list = None
svd_scale = get_svd_scale(model, tokenizer, args.model_type, args.dataset_name, args.split, args.seed, args.seqlen, args.batch_size, args.max_samples, selected_layers_list)
if selected_layers_list is not None:
    selected_layers_str = '_'.join(map(str, selected_layers_list))
else:
    selected_layers_str = "all"

torch.save(svd_scale, f"{args.save_path}/SVD_scale_{args.model_type}_{selected_layers_str}_{args.max_samples}.pt")
