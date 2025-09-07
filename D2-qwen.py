import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset, make_batchnorm_stats, make_calibration_dataloader
from metric import make_metric, make_logger
from model import make_prune_model
from module import to_device, process_control, makedir_exist_ok, check_dense_model, save_calib_info
from deepspeed.profiling.flops_profiler import FlopsProfiler
from utils import run_lm_eval, ppl_eval_sharing
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from model.merge_qwen import Merge_QwenMoE



cudnn.benchmark = False
parser = argparse.ArgumentParser(description='cfg')
parser.add_argument('--base_model_path', type=str, default="Qwen/Qwen2-57B-A14B", help='Path to base model')
parser.add_argument('--expert_freq_path', type=str, default="cache/QwenMoE_wikitext_20000_expert_frequencies.json", help='Path to expert frequencies')
parser.add_argument('--fisher_path', type=str, default="Model/fisher_QwenMoE.pt", help='Path to fisher info')
parser.add_argument('--svd_scale_path', type=str, default="Model/SVD_scale_QwenMoE_0-31_512.pt", help='Path to svd scale')
parser.add_argument('--result_path', type=str, default="result", help='Path to result')

parser.add_argument("--pp_ratio", type=float, default=0.2)
parser.add_argument("--delta_ratio", type=float, default=1)
parser.add_argument("--share_ratio", type=float, default=1)
parser.add_argument("--share_V", action='store_true', default=False)
parser.add_argument("--share_U", action='store_true', default=False)
parser.add_argument("--merge_method", type=str, default="fisher")


for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
parser.add_argument('--output_dir', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    cfg['prune_ratio'] = args['pp_ratio']
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    result_path = os.path.join('output', 'result')
    makedir_exist_ok(result_path)
    if check_dense_model() is None:
        pass
    cfg['epoch'] = 0
    cfg['data_name'] = 'wikitext'

    dataset = make_dataset(cfg['data_name'], cfg['subset_name'])
    cfg['model_name'] = 'mixtral'
    cfg['skip_layers'] = []
    cfg['test_stage'] = False
    cfg['no_probe_process'] = False

    cfg['merge_model'] = True

    cfg['shared_infer'] = False




    def merge_model(base_model_path, expert_freq_path, svd_scale_path, fisher_path, delta_ratio, share_ratio, share_V, share_U, merge_method):
        import json
        with open(expert_freq_path, 'r') as f:
            expert_freq = json.load(f)
        svd_scale = torch.load(svd_scale_path, map_location='cpu')
        fisher_info = torch.load(fisher_path, map_location="cpu")

        model = AutoModelForCausalLM.from_pretrained(base_model_path, 
                                                    device_map="auto", 
                                                    trust_remote_code=True, 
                                                    torch_dtype=torch.bfloat16)

        print("Baseline model loaded.")

        # ==================== ⬇️ 新增的即时扫描代码 ⬇️ ====================
        def verify_loaded_model(model_to_check):
            print("\n" + "="*60)
            print("🔬 Performing immediate scan of the loaded model in memory...")
            print("="*60)
            
            found_issue = False
            corrupted_tensors = []
            
            pbar = tqdm(model_to_check.named_parameters(), desc="Scanning loaded parameters")
            for name, param in pbar:
                if not torch.all(torch.isfinite(param)):
                    nan_count = torch.isnan(param).sum().item()
                    inf_count = torch.isinf(param).sum().item()
                    issue_str = f"Tensor: {name}, NaNs: {nan_count}, Infs: {inf_count}"
                    corrupted_tensors.append(issue_str)
                    found_issue = True

            if not found_issue:
                print("\n✅ VERDICT: Model in memory is CLEAN. No NaN/Inf values found immediately after loading.")
            else:
                print(f"\n❌ VERDICT: CORRUPTION DETECTED immediately after loading!")
                print("The following tensors were found to be corrupted:")
                for issue in corrupted_tensors:
                    print(f"  - {issue}")
                # 如果检测到损坏，可以选择直接抛出异常停止程序
                # raise RuntimeError("Model is corrupted upon loading, stopping execution.")
                
            print("="*60 + "\n")

        # 执行扫描
        # verify_loaded_model(model)
        # ==================== ⬆️ 扫描代码结束 ⬆️ ====================



        # ==================== ⬇️ 开始修改 ⬇️ ====================
    
        # 1. 定义一个只包含你想要处理的层的列表
        # layers_to_process = [12, 13, 14, 16]
        # print(f"Targeting specific layers for merging: {layers_to_process}")

        # 2. 修改 for 循环，让它只遍历上面这个列表中的层号
        # for i in tqdm(layers_to_process, desc="Merging specific layers"):
        # ==================== ⬆️ 修改结束 ⬆️ ====================

        for i in tqdm(range(len(model.model.layers)), desc="Merging layers"):
            try:
                Merge_MoE_Block = Merge_QwenMoE(model.config, share_ratio=share_ratio, 
                                                        delta_ratio=delta_ratio, expert_freq=expert_freq[str(i)], 
                                                        delta_share_V=share_V, delta_share_U=share_U, 
                                                        merge_method=merge_method, shared_infer=cfg['shared_infer']).to(model.model.layers[i].mlp.gate.weight.device)
                Merge_MoE_Block.merge_experts(model.model.layers[i].mlp, svd_scale=svd_scale[i], hessian = fisher_info[i], scale_type='svdllm')
                model.model.layers[i].mlp = Merge_MoE_Block
            except ValueError as e:
                print(f"Warning: SVD failed for layer {i}, skipping this layer")
                continue
        
        return model

    model = merge_model(args['base_model_path'], args['expert_freq_path'], args['svd_scale_path'], args['fisher_path'], 
                        args['delta_ratio'], args['share_ratio'], args['share_V'], args['share_U'], args['merge_method'])
    tokenizer = AutoTokenizer.from_pretrained(args['base_model_path'], use_fast=False)


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # -----------------------------------------------------------------------------------------
    cfg['tokenizer'] = tokenizer
    cfg['model_name'] = 'llama-2-7b'
    cfg['model_type'] = 'qwen'

    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    dataset = process_dataset(dataset, tokenizer)



    if cfg['model_name'] in ['cnn', 'resnet18', 'wresnet28x2']:
        model = make_batchnorm_stats(dataset['train'], model, cfg['model_name'])
    
    model = make_prune_model(model)
    
    # if cfg['merge_model']:
    #     for i in range(len(model.model.model.layers)):
    #         model.model.model.layers[i].mlp.update_Wmean()

    # 修改后 (添加了安全检查)
    if cfg['merge_model']:
        print("Updating Wmean for merged layers...")
        for i in range(len(model.model.model.layers)):
            # 检查当前层的 mlp 模块是否拥有 update_Wmean 这个方法
            if hasattr(model.model.model.layers[i].mlp, 'update_Wmean'):
                # 如果有，才调用它
                model.model.model.layers[i].mlp.update_Wmean()
            else:
                # 如果没有，说明是原始模块，打印信息并跳过
                print(f"Skipping update_Wmean for layer {i} (original MoE block).")

    if 'calib' in cfg['prune_method']:
        print('Running Calibration ...', flush=True)
        cfg['calibration_stage'] = True
        cfg['calibration_dataset'] = 'wikitest'
        calibration_data_loader = make_calibration_dataloader(tokenizer)
        run_calibration(model, calibration_data_loader['train'])
        save_calib_info(model)
        if 'flapratio' in cfg['prune_method']:
            from model import HiddenRepresentationPruning
            pruning_module = HiddenRepresentationPruning(cfg, 'flapratio')
            pruning_module.flap_ratio(model, test_logger)
        cfg['calibration_stage'] = False
        print('Calibration Done...', flush=True)


    save_dir = f"{args['result_path']}/qwen_delta-{args['delta_ratio']}-pp_ratio-{args['pp_ratio']}-shareV-{args['share_V']}"
    os.makedirs(save_dir, exist_ok=True)

    # result_str = ppl_eval_sharing(model, tokenizer, experiment_name=f"D2-qwen", datasets=['wikitext2', 'ptb', 'c4'], params_only=False, batch_size=8)
    # with open(f"{save_dir}/ppl_eval_sharing.txt", "w") as f:
    #     f.write(result_str)

    # run_lm_eval(model, tokenizer, batch_size=8, task_names=["openbookqa", "arc_easy", "winogrande", "hellaswag",
    #         "arc_challenge", "piqa", "mathqa"], output_dir=save_dir)
    
    # ===================================================================
    #                  ✅ 开始：标准延迟测试代码块 ✅
    # ===================================================================
    import time
    import numpy as np
    import json

    print("\n" + "="*50)
    print(" " * 15 + "Running Latency Test")
    print("="*50)

    # --- 1. 参数设置 (与之前保持一致) ---
    prompt = "DeepSeek is a large language model developed by"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) 
    n_warmup = 5
    n_runs = 20
    max_new_tokens = 100
    latencies = []

    # --- 2. 预热 (Warm-up) ---
    print(f"Running {n_warmup} warm-up rounds...")
    for _ in range(n_warmup):
        _ = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    torch.cuda.synchronize()

    # --- 3. 正式测试 ---
    print(f"Running {n_runs} test rounds...")
    for _ in tqdm(range(n_runs), desc="Measuring Latency"):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        _ = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        latencies.append(end_time - start_time)

    # --- 4. 计算、打印并保存结果 ---
    avg_latency_per_run = np.mean(latencies)
    avg_latency_per_token = (avg_latency_per_run / max_new_tokens) * 1000
    tokens_per_second = 1 / (avg_latency_per_run / max_new_tokens)

    print("\n" + "="*50)
    print(" " * 17 + "Latency Test Results")
    print("="*50)
    print(f"Model: {args['base_model_path']}")
    print(f"Test runs: {n_runs}")
    print(f"Generated tokens per run: {max_new_tokens}\n")
    print(f"➡️ Average latency per run: {avg_latency_per_run:.4f} seconds")
    print(f"➡️ Average latency per token: {avg_latency_per_token:.2f} ms/token")
    print(f"➡️ Throughput: {tokens_per_second:.2f} tokens/second")
    print("="*50 + "\n")

    latency_results = {
        "experiment_parameters": {
            "base_model": args['base_model_path'],
            "merge_method": args['merge_method'],
            "pp_ratio": args['pp_ratio'],
            "delta_ratio": args['delta_ratio'],
            "share_V": args['share_V']
        },
        "latency_test_config": {
            "prompt": prompt,
            "warmup_runs": n_warmup,
            "test_runs": n_runs,
            "max_new_tokens": max_new_tokens
        },
        "results": {
            "avg_latency_per_run_s": round(avg_latency_per_run, 4),
            "avg_latency_per_token_ms": round(avg_latency_per_token, 2),
            "throughput_tokens_per_sec": round(tokens_per_second, 2)
        }
    }
    
    latency_json_path = os.path.join(save_dir, "latency_results.json")
    with open(latency_json_path, "w") as f:
        json.dump(latency_results, f, indent=4)
    print(f"Latency results saved to: {latency_json_path}")
    # ===================================================================
    #                   ✅ 结束：标准延迟测试代码块 ✅
    # ===================================================================
    
    return


def run_calibration(model, data_loader):    
    with torch.no_grad():
        model.eval()
        for i, input in enumerate(data_loader):
            # now, the wikitext and c4 datsets used for calibration are clm tasks
            # input_size = input['labels'].size(0)
            input = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
                    'labels': input['labels']}
            input = to_device(input, "cuda")
            output = model(**input)
            # input_ = {'target': input['labels']}
            # output_ = {'target': output['logits'], 'loss': output['loss']}
    return





if __name__ == "__main__":
    main()

