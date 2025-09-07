import argparse
import os
import glob
import torch
from tqdm import tqdm

try:
    import safetensors.torch
except ImportError:
    print("Error: `safetensors` library not found. Please run `pip install safetensors`.")
    exit()

def check_tensors_in_state_dict(state_dict, device: str):
    """辅助函数：检查一个state_dict中的所有张量是否有NaN/Inf。"""
    for key, tensor in state_dict.items():
        if not torch.all(torch.isfinite(tensor)):
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            return key, nan_count, inf_count
    return None, 0, 0

def verify_model_shards_incrementally(model_path: str):
    """
    以分片方式、内存高效地扫描模型文件，并在CPU和GPU上分别检查。
    """
    print("="*60)
    print(f"🔬 Starting incremental verification for model at: {model_path}")
    print("="*60)

    if not os.path.isdir(model_path):
        print(f"Error: Directory not found at '{model_path}'")
        return

    checkpoint_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not checkpoint_files:
        print("Error: No .safetensors files found.")
        return

    corrupted_stages = []
    
    # 检查是否有可用的CUDA设备
    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        print("Warning: No CUDA device found. Will only perform CPU checks.")

    for filepath in tqdm(checkpoint_files, desc="Scanning Shards"):
        filename = os.path.basename(filepath)

        # --- 1. CPU 检查 ---
        try:
            state_dict_cpu = safetensors.torch.load_file(filepath, device="cpu")
            corrupted_key, nans, infs = check_tensors_in_state_dict(state_dict_cpu, "cpu")
            if corrupted_key:
                issue = f"File: {filename}, Stage: CPU Loading, Tensor: {corrupted_key}, NaNs: {nans}, Infs: {infs}"
                corrupted_stages.append(issue)
                # tqdm.write(f"❌ Corruption detected on CPU for {filename}")
                continue # 如果CPU加载就有问题，没必要再检查GPU
        except Exception as e:
            issue = f"File: {filename}, Stage: CPU Loading, Error: {e}"
            corrupted_stages.append(issue)
            # tqdm.write(f"❌ Error loading {filename} to CPU")
            continue

        # --- 2. GPU 检查 (如果可用) ---
        if use_gpu:
            try:
                # 为了节省内存，先删除CPU上的副本
                del state_dict_cpu
                torch.cuda.empty_cache()

                state_dict_gpu = safetensors.torch.load_file(filepath, device="cuda:0")
                corrupted_key, nans, infs = check_tensors_in_state_dict(state_dict_gpu, "cuda:0")
                if corrupted_key:
                    issue = f"File: {filename}, Stage: GPU Loading, Tensor: {corrupted_key}, NaNs: {nans}, Infs: {infs}"
                    corrupted_stages.append(issue)
                    # tqdm.write(f"❌ Corruption detected on GPU for {filename}")
                
                # 清理显存
                del state_dict_gpu
                torch.cuda.empty_cache()

            except Exception as e:
                issue = f"File: {filename}, Stage: GPU Loading, Error: {e}"
                corrupted_stages.append(issue)
                # tqdm.write(f"❌ Error loading {filename} to GPU")

    # --- 最终报告 ---
    print("\n" + "="*60)
    print(" " * 18 + "Final Verification Report")
    print("="*60)
    
    if not corrupted_stages:
        print("✅ Verdict: The model appears to be CLEAN on both CPU and GPU.")
        print("No corruption was detected during the loading process.")
    else:
        print("❌ Verdict: CORRUPTION DETECTED during the loading process!")
        print("Issues were found at the following stages:")
        for issue in corrupted_stages:
            print(f"  - {issue}")
        print("\nRecommendation: The problem likely lies in the CUDA driver, PyTorch environment, or hardware during CPU->GPU data transfer.")
    
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify model shards on CPU and GPU.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory.")
    args = parser.parse_args()
    verify_model_shards_incrementally(args.model_path)