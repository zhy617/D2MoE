import argparse
import os
import glob
import torch
from tqdm import tqdm

# 尝试导入 safetensors，如果失败则提示用户安装
try:
    import safetensors.torch
except ImportError:
    print("Warning: `safetensors` library not found. Please install it with `pip install safetensors` to check .safetensors files.")
    safetensors = None

def verify_model_files(model_path: str):
    """
    Scans all model weight files (.safetensors, .bin) in a directory
    for NaN and Inf values in a memory-efficient way.
    """
    print("="*60)
    print(f"🔬 Starting verification for model at: {model_path}")
    print("="*60)

    if not os.path.isdir(model_path):
        print(f"Error: Directory not found at '{model_path}'")
        return

    # 查找所有 safetensors 和 bin 格式的权重文件
    checkpoint_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    checkpoint_files += glob.glob(os.path.join(model_path, "*.bin"))

    if not checkpoint_files:
        print("Error: No model checkpoint files (.safetensors or .bin) found in the directory.")
        return

    total_tensors_scanned = 0
    corrupted_tensors = []

    for filepath in tqdm(checkpoint_files, desc="Scanning files"):
        filename = os.path.basename(filepath)
        file_had_issue = False
        
        try:
            if filepath.endswith(".safetensors"):
                if safetensors is None:
                    # tqdm.write(f"Skipping {filename} as safetensors library is not installed.")
                    continue
                
                # 使用 safetensors 的安全打开模式，逐个加载张量，非常节省内存
                with safetensors.torch.safe_open(filepath, framework="pt", device="cpu") as f:
                    for key in tqdm(f.keys(), desc=f"  -> {filename}", leave=False):
                        tensor = f.get_tensor(key)
                        total_tensors_scanned += 1
                        
                        has_nan = torch.isnan(tensor).any()
                        has_inf = torch.isinf(tensor).any()

                        if has_nan or has_inf:
                            corrupted_tensors.append(f"File: {filename}, Tensor: {key}")
                            file_had_issue = True

            elif filepath.endswith(".bin"):
                # torch.load 会一次性加载整个文件，对内存有一定要求
                state_dict = torch.load(filepath, map_location="cpu")
                for key, tensor in tqdm(state_dict.items(), desc=f"  -> {filename}", leave=False):
                    total_tensors_scanned += 1
                    
                    has_nan = torch.isnan(tensor).any()
                    has_inf = torch.isinf(tensor).any()

                    if has_nan or has_inf:
                        corrupted_tensors.append(f"File: {filename}, Tensor: {key}")
                        file_had_issue = True

        except Exception as e:
            print(f"\n🚨 Error while processing file {filename}: {e}")
            corrupted_tensors.append(f"File: {filename}, Error: COULD NOT BE READ. Possibly corrupted.")

    print("\n" + "="*60)
    print(" " * 22 + "Verification Report")
    print("="*60)
    print(f"Total files scanned: {len(checkpoint_files)}")
    print(f"Total tensors scanned: {total_tensors_scanned}")

    if not corrupted_tensors:
        print("\n✅ Verdict: The model files appear to be CLEAN.")
        print("No NaN or Inf values were found.")
    else:
        print(f"\n❌ Verdict: CORRUPTION DETECTED!")
        print(f"Found {len(corrupted_tensors)} tensor(s) with NaN/Inf values or read errors:")
        for issue in corrupted_tensors:
            print(f"  - {issue}")
        print("\nRecommendation: Delete the local model files and download a fresh copy.")
    
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Hugging Face model files for NaN/Inf corruption.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the local directory containing the model files.")
    args = parser.parse_args()
    
    verify_model_files(args.model_path)

# verify-model.py
# --model_path=/home/tom/fsas/models/Qwen/Qwen2-57B-A14B