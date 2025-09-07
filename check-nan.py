import torch

# !!! 请将这里的路径替换为您自己的文件路径 !!!
file_path = "/home/tom/fsas/zhanghongyu/results/qwen/SVD_scale_qwen_all_256.pt"

print(f"Checking for NaNs in file: {file_path}")

try:
    data = torch.load(file_path, map_location='cpu')
    
    found_nan = False
    # data 可能是字典或列表，我们遍历其中的所有张量
    if isinstance(data, dict):
        for key, tensor in data.items():
            if torch.is_tensor(tensor) and torch.isnan(tensor).any():
                print(f"  -> CRITICAL: NaN found in tensor with key '{key}'!")
                found_nan = True
    elif isinstance(data, list):
        for i, tensor in enumerate(data):
            if torch.is_tensor(tensor) and torch.isnan(tensor).any():
                print(f"  -> CRITICAL: NaN found in tensor at index {i}!")
                found_nan = True

    if not found_nan:
        print("  -> OK: No NaNs found in the file.")

except Exception as e:
    print(f"An error occurred while reading the file: {e}")