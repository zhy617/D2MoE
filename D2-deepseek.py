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
from model.merge_deepseek import Merge_deepseekMoE



cudnn.benchmark = False
parser = argparse.ArgumentParser(description='cfg')
parser.add_argument('--base_model_path', type=str, default="deepseek-ai/deepseek-moe-16b-base", help='Path to base model')
parser.add_argument('--expert_freq_path', type=str, default="cache/deepseek-moe-16b-base_wikitext_20000_expert_frequencies.json", help='Path to expert frequencies')
parser.add_argument('--fisher_path', type=str, default="Model/fisher_deepseek-moe-16b-base.pt", help='Path to fisher info')
parser.add_argument('--svd_scale_path', type=str, default="Model/SVD_scale_deepseek-moe-16b-base_0-31_512.pt", help='Path to svd scale')
parser.add_argument('--result_path', type=str, default="result", help='Path to result')

parser.add_argument("--pp_ratio", type=float, default=0.2)
parser.add_argument("--delta_ratio", type=float, default=1)
parser.add_argument("--share_ratio", type=float, default=1)
parser.add_argument("--share_V", action='store_true', default=False)
parser.add_argument("--share_U", action='store_true', default=False)
parser.add_argument("--merge_method", type=str, default="fisher")
# 在这里添加新的参数
# parser.add_argument('--probe_ratio_my_set', type=float, default=0.1, help='The probe ratio for pruning metric calculation.')


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


        for i in tqdm(range(len(model.model.layers)), desc="Merging layers"):
            if i == 0:
                # skip mlp layer
                continue
            Merge_MoE_Block = Merge_deepseekMoE(model.config, share_ratio=share_ratio, 
                                                        delta_ratio=delta_ratio, expert_freq=expert_freq[str(i)], 
                                                        delta_share_V=share_V, delta_share_U=share_U, 
                                                        merge_method=merge_method, shared_infer=cfg['shared_infer']).to(model.model.layers[i].mlp.gate.weight.device)
            Merge_MoE_Block.merge_experts(model.model.layers[i].mlp, svd_scale=svd_scale[i], hessian = fisher_info[i], scale_type='svdllm')
            model.model.layers[i].mlp = Merge_MoE_Block
        
        return model

    model = merge_model(args['base_model_path'], args['expert_freq_path'], args['svd_scale_path'], args['fisher_path'], 
                        args['delta_ratio'], args['share_ratio'], args['share_V'], args['share_U'], args['merge_method'])
    tokenizer = AutoTokenizer.from_pretrained(args['base_model_path'], use_fast=False)


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # -----------------------------------------------------------------------------------------
    cfg['tokenizer'] = tokenizer
    cfg['model_name'] = 'llama-2-7b'
    cfg['model_type'] = 'deepseek'

    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    dataset = process_dataset(dataset, tokenizer)



    if cfg['model_name'] in ['cnn', 'resnet18', 'wresnet28x2']:
        model = make_batchnorm_stats(dataset['train'], model, cfg['model_name'])
    
    model = make_prune_model(model)
    
    if cfg['merge_model']:
        for i in range(len(model.model.model.layers)):
            if i == 0:
                continue
            model.model.model.layers[i].mlp.update_Wmean()

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


    save_dir = f"{args['result_path']}/deepseek_delta-{args['delta_ratio']}-pp_ratio-{args['pp_ratio']}-shareV-{args['share_V']}"
    os.makedirs(save_dir, exist_ok=True)

    # result_str = ppl_eval_sharing(model, tokenizer, experiment_name=f"D2-deepseek", datasets=['wikitext2', 'ptb', 'c4'], params_only=False, batch_size=16)
    # with open(f"{save_dir}/ppl_eval_sharing.txt", "w") as f:
    #     f.write(result_str)

    # run_lm_eval(model, tokenizer, batch_size=16, task_names=["openbookqa", "arc_easy", "winogrande", "hellaswag",
    #         "arc_challenge", "piqa", "mathqa"], output_dir=save_dir)
    
     # ===================================================================
    #                  ⬇️ 在这里添加下面的延迟测试代码 ⬇️
    # ===================================================================
    import time
    import numpy as np

    print("\n" + "="*50)
    print(" " * 15 + "Running Latency Test")
    print("="*50)

    # --- 1. 参数设置 ---
    prompt = "DeepSeek is a large language model developed by" # 用于测试的输入文本
    # input_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    # 修改点 1：接收完整的 tokenizer 输出 (inputs 字典)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) 
    n_warmup = 5      # 预热轮数
    n_runs = 20       # 实际测试轮数
    max_new_tokens = 100 # 每次生成的目标 token 数量

    latencies = []

    # --- 2. 预热 (Warm-up) ---
    # 第一次推理会包含一些额外的编译开销，预热可以排除这些干扰，让测试结果更准确
    print(f"Running {n_warmup} warm-up rounds...")
    for _ in range(n_warmup):
        _ = model.generate(input_tokens, max_new_tokens=max_new_tokens, do_sample=False)
    
    torch.cuda.synchronize() # 确保所有预热操作在 GPU 上已完成

    # --- 3. 正式测试 ---
    print(f"Running {n_runs} test rounds...")
    for _ in tqdm(range(n_runs), desc="Measuring Latency"):
        torch.cuda.synchronize() # 同步，确保计时器起点准确
        start_time = time.perf_counter()
        
        # 核心推理步骤
        # generated_ids = model.generate(input_tokens, max_new_tokens=max_new_tokens, do_sample=False)
        
        
        torch.cuda.synchronize() # 同步，确保计时器终点准确
        end_time = time.perf_counter()
        
        # 记录本次运行的时间
        latencies.append(end_time - start_time)

    # --- 4. 计算、打印结果并写入JSON文件 ---
    import json

    total_tokens_generated = n_runs * max_new_tokens
    total_time_spent = sum(latencies)

    avg_latency_per_run = np.mean(latencies)
    avg_latency_per_token = (avg_latency_per_run / max_new_tokens) * 1000  # 转换为毫秒
    tokens_per_second = 1 / (avg_latency_per_run / max_new_tokens)

    # --- 在控制台打印结果 (与之前相同) ---
    print("\n" + "="*50)
    print(" " * 17 + "Latency Test Results")
    print("="*50)
    print(f"Model: {args['base_model_path']}")
    print(f"Test runs: {n_runs}")
    print(f"Generated tokens per run: {max_new_tokens}\n")
    print(f"➡️ Average latency per run (generating {max_new_tokens} tokens): {avg_latency_per_run:.4f} seconds")
    print(f"➡️ Average latency per token: {avg_latency_per_token:.2f} ms/token")
    print(f"➡️ Throughput: {tokens_per_second:.2f} tokens/second")
    print("="*50 + "\n")


    # --- 将关键参数和结果组织成一个字典 ---
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

    # --- 将字典写入 JSON 文件 ---
    # save_dir 是你脚本前面已经定义好的结果保存目录
    latency_json_path = os.path.join(save_dir, "latency_results.json")
    with open(latency_json_path, "w") as f:
        json.dump(latency_results, f, indent=4)

    print(f"Latency results saved to: {latency_json_path}")
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

