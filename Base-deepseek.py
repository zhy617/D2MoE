import argparse
import os
import torch
import numpy as np
import time
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def measure_baseline_latency(args):
    """
    加载一个指定的Hugging Face模型并测试其生成延迟。
    """
    print("="*50)
    print(" " * 10 + "Starting Baseline Latency Test")
    print("="*50)
    print(f"Model Path: {args.model_path}")

    # 1. 加载基线模型和分词器
    # ----------------------------------------------------
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",          # 自动将模型分布到可用GPU
        torch_dtype=torch.bfloat16, # 使用bfloat16以获得更好的性能
        trust_remote_code=True      # 对于某些模型是必需的
    )
    model.eval() # 设置为评估模式

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 准备输入
    # ----------------------------------------------------
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    latencies = []

    # 3. 预热 (Warm-up)
    # ----------------------------------------------------
    print(f"\nRunning {args.n_warmup} warm-up rounds...")
    for _ in range(args.n_warmup):
        _ = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    torch.cuda.synchronize()

    # 4. 正式测试
    # ----------------------------------------------------
    print(f"\nRunning {args.n_runs} test rounds...")
    for _ in tqdm(range(args.n_runs), desc="Measuring Latency"):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        _ = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        latencies.append(end_time - start_time)

    # 5. 计算、打印并保存结果
    # ----------------------------------------------------
    avg_latency_per_run = np.mean(latencies)
    avg_latency_per_token = (avg_latency_per_run / args.max_new_tokens) * 1000
    tokens_per_second = 1 / (avg_latency_per_run / args.max_new_tokens)

    # 在控制台打印
    print("\n" + "="*50)
    print(" " * 14 + "Baseline Latency Results")
    print("="*50)
    print(f"Model: {args.model_path}")
    print(f"Test runs: {args.n_runs}")
    print(f"Generated tokens per run: {args.max_new_tokens}\n")
    print(f"➡️ Average latency per run: {avg_latency_per_run:.4f} seconds")
    print(f"➡️ Average latency per token: {avg_latency_per_token:.2f} ms/token")
    print(f"➡️ Throughput: {tokens_per_second:.2f} tokens/second")
    print("="*50 + "\n")

    # 组织结果并保存为JSON
    results = {
        "model_path": args.model_path,
        "test_config": {
            "prompt": args.prompt,
            "warmup_runs": args.n_warmup,
            "test_runs": args.n_runs,
            "max_new_tokens": args.max_new_tokens
        },
        "results": {
            "avg_latency_per_run_s": round(avg_latency_per_run, 4),
            "avg_latency_per_token_ms": round(avg_latency_per_token, 2),
            "throughput_tokens_per_sec": round(tokens_per_second, 2)
        }
    }
    
    os.makedirs(args.save_path, exist_ok=True)
    # 从模型路径中提取模型名称作为文件名
    model_name = os.path.basename(args.model_path.strip('/'))
    json_path = os.path.join(args.save_path, f"baseline_latency_{model_name}.json")
    
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Baseline latency results saved to: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Model Latency Test")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the baseline model on Hugging Face or local disk.')
    parser.add_argument('--save_path', type=str, default="results_baseline", help='Directory to save the results JSON file.')
    parser.add_argument('--prompt', type=str, default="DeepSeek is a large language model developed by", help='Input prompt for the generation test.')
    parser.add_argument('--n_warmup', type=int, default=5, help='Number of warm-up rounds.')
    parser.add_argument('--n_runs', type=int, default=20, help='Number of timed test rounds.')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Number of new tokens to generate in each run.')
    
    args = parser.parse_args()
    measure_baseline_latency(args)