from dataclasses import dataclass, field
from typing import Optional, List, Literal
import torch
import time
from dataset import collate
from config import cfg
from module import model_forward, update_model_prof, to_device
from datetime import datetime
from tqdm import tqdm
from transformers import TrainingArguments
from torch.nn.utils.rnn import pad_sequence
from peft import PeftModel, LoraConfig, get_peft_model
import numpy as np
import transformers

def print_memory_usage():
    total_gpus = torch.cuda.device_count()
    total_allocated = 0
    total_reserved = 0
    
    for i in range(total_gpus):
        allocated = torch.cuda.memory_allocated(device=i) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(device=i) / 1024 / 1024
        total_allocated += allocated
        total_reserved += reserved
        # print(f"GPU {i} - Allocated: {allocated:.2f} MiB, Reserved: {reserved:.2f} MiB")
    
    # print(f"Total - Allocated: {total_allocated:.2f} MiB, Reserved: {total_reserved:.2f} MiB")
    
    return total_allocated, total_reserved


@torch.no_grad()
def ppl_eval_sharing(model, tokenizer, experiment_name, datasets=['wikitext2', 'ptb', 'c4'], model_seq_len=2048, batch_size=16, params_only=False):
    seed = 42  # or any other integer
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    def _perplexity(nlls, n_samples, seqlen):
        return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))

    model.eval()
    ppls = {}
    total_allocated_list = []
    total_reserved_list = []

    # main_device = next(model.parameters()).device
    main_device = list(model.parameters())[-1].device
    if not params_only:
        for dataset in datasets:
            # 对于其他数据集，使用原有的加载方式
            data = get_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size=batch_size)

            seqlen = model_seq_len
            n_samples = len(data)
            nlls = []

            with tqdm(range(n_samples), desc=f"Evaluating {dataset} - Perplexity") as progress_bar:
                for i in progress_bar:
                    batch = next(iter(data)).to(main_device)

                    allocated, reserved = print_memory_usage()
                    total_allocated_list.append(allocated)
                    total_reserved_list.append(reserved)

                    with torch.no_grad():
                        output = model(batch)
                        logits = output.logits if hasattr(output, "logits") else output[0]

                    # 确保 logits 在正确的设备上
                    logits = logits.to(main_device)
                    shift_logits = logits[:, :-1, :].contiguous().float()
                    shift_labels = batch[:, 1:].contiguous()

                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    neg_log_likelihood = loss.float() * seqlen
                    nlls.append(neg_log_likelihood)

                    curr_ppl = _perplexity(nlls, i + 1, seqlen)
                    progress_bar.set_description(f"Evaluating {dataset} - Perplexity {curr_ppl:.3f}")

            ppl = _perplexity(nlls, n_samples, seqlen)
            ppls[dataset] = ppl.item()


    # 计算参数统计
    total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # non_trainable_params = total_params - trainable_params
    threshold = 1e-6
    non_zero_params = sum((p.abs() > threshold).sum().item() for p in model.parameters())

    # 检查 SVD 压缩的 Mixtral 特性
    # svd_layers = sum(1 for m in model.modules() if isinstance(m, SVD_MixtralSparseMoeBlock))
    print("\n")
    result_str = f"Experiment: {experiment_name}\n"
    if not params_only:
        avg_allocated = sum(total_allocated_list) / len(total_allocated_list)
        avg_reserved = sum(total_reserved_list) / len(total_reserved_list)
        result_str += f"PPL after evaluation: {ppls}\n"
        result_str += f"Average Allocated Memory: {avg_allocated:.2f} MiB\n"
        result_str += f"Average Reserved Memory: {avg_reserved:.2f} MiB\n"
    
    result_str += f"Total number of parameters: {total_params / 1e9:.2f}B\n"
    # result_str += f"Number of trainable parameters: {trainable_params / 1e9:.2f}B\n"
    # result_str += f"Number of non-trainable parameters: {non_trainable_params / 1e9:.2f}B\n"
    result_str += f"Number of non-zero parameters: {non_zero_params / 1e9:.2f}B\n"
    if "Mixtral" in experiment_name:
        org_params = 46.70e9
        result_str += f"Compression ratio: {1 - (total_params / org_params):.2f}%\n"
        result_str += f"Save ratio: {total_params / org_params:.2f}%\n"

    elif "Llamix" in experiment_name:
        org_params = 0.40e9
        result_str += f"Compression ratio: {1 - (total_params / org_params):.2f}%\n"
        result_str += f"Save ratio: {total_params / org_params:.2f}%\n"

    elif "PhiMoE" in experiment_name:
        org_params = 41.83e9
        result_str += f"Compression ratio: {1 - (total_params / org_params):.2f}%\n"
        result_str += f"Save ratio: {total_params / org_params:.2f}%\n"
    
    elif "deepseek" in experiment_name:
        org_params = 16.38e9
        result_str += f"Compression ratio: {1 - (total_params / org_params):.2f}%\n"
        result_str += f"Save ratio: {total_params / org_params:.2f}%\n"

    elif "Qwen" in experiment_name or "qwen" in experiment_name:
        org_params = 57.41e9
        result_str += f"Compression ratio: {1 - (total_params / org_params):.2f}%\n"
        result_str += f"Save ratio: {total_params / org_params:.2f}%\n"
    else:
        pass

    print(result_str)
    return result_str

def identify_pad_tokens(input):
    pad_tokens = input['input_ids'] == cfg['tokenizer'].pad_token if cfg['tokenizer'].pad_token is not None else cfg['tokenizer'].eos_token
    if isinstance(pad_tokens, int):
        pad_tokens = torch.tensor(pad_tokens)
    no_padding = (~pad_tokens).all()
    # if there is padding, need to zero out the padding token
    if no_padding == False:
        cfg['pad_tokens'] = pad_tokens
        # cfg['non_pad_tokens'] = ~pad_tokens.to(cfg['data_type'])
        # avoid overflow
        cfg['nonpad_tokens_denominator'] = torch.sum(~cfg['pad_tokens'], dim=0).unsqueeze(1) + 1e-3
    else:
        cfg['pad_tokens'] = None
        # cfg['non_pad_tokens'] = None
        cfg['nonpad_tokens_denominator'] = None
    return

def test(data_loader, model, model_prof, metric, logger):
    torch.cuda.empty_cache()
    start_time = time.time()

    with torch.no_grad():
        
        model.train(False)
        start_time = time.time()
        inference_duration = 0

        # warm up pytorch
        data_loader_iter = iter(data_loader)
        input = next(data_loader_iter)
        identify_pad_tokens(input)
        # print('start input_ids', input['input_ids'], input['input_ids'].size())
        cfg['cur_batch_index'] += 1
        if cfg['task_name'] in ['clm']:
            input_size = input['labels'].size(0)
            input = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
                    'labels': input['labels']}
            input = to_device(input, cfg['device'])
            output = model(**input)
            input_ = {'target': input['labels']}
            output_ = {'target': output['logits'], 'loss': output['loss']}
        elif cfg['task_name'] in ['csr']:
            input_size = input['labels'].size(0)
            input_indices = input['input_indices']
            correct_labels = input['correct_labels']
            input = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
                    'labels': input['labels']}
            input = to_device(input, cfg['device'])
            output = model(**input)
            input_ = {'input_indices': input_indices, 'target': input['labels'], 'correct_labels': correct_labels}
            output_ = {'target': output['logits'], 'loss': output['loss']}
        else:
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(**input)
            input_ = {'target': input['target']}
            output_ = {'target': output['target'], 'loss': output['loss']}
        torch.cuda.synchronize()


        inference_duration_list = []

        model_prof.start_profile()
        model_prof.reset_profile()
        update_model_prof(model_prof)
        torch.cuda.cudart().cudaProfilerStart()
        for i, input in enumerate(data_loader):
            cfg['cur_batch_index'] += 1
            identify_pad_tokens(input)
            if cfg['task_name'] in ['s2s', 'sc', 'clm']:
                input_size = input['labels'].size(0)
                input = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
                        'labels': input['labels']}
                input = to_device(input, cfg['device'])
                output, inference_duration, cur_inference_duration = model_forward(model, input, inference_duration, i)
                inference_duration_list.append(cur_inference_duration)
                input_ = {'target': input['labels']}
                output_ = {'target': output['logits'], 'loss': output['loss']}
            elif cfg['task_name'] in ['csr']:
                input_size = input['labels'].size(0)
                input_indices = input['input_indices']
                correct_labels = input['correct_labels']
                # print('input', input)
                input = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
                        'labels': input['labels']}
                input = to_device(input, cfg['device'])
                output, inference_duration, cur_inference_duration = model_forward(model, input, inference_duration, i)
                inference_duration_list.append(cur_inference_duration)
                input_ = {'input_indices': input_indices, 'target': input['labels'], 'correct_labels': correct_labels}
                output_ = {'target': output['logits'], 'loss': output['loss']}
            else:
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                output, inference_duration, cur_inference_duration = model_forward(model, input, inference_duration, i)
                inference_duration_list.append(cur_inference_duration)
                input_ = {'target': input['target']}
                output_ = {'target': output['target'], 'loss': output['loss']}

            if cfg['onlyprobe'] == False: 
                metric.add('test', input_, output_)
                evaluation = metric.evaluate('test', 'batch', input_, output_)
                print('evaluation_for_batch', evaluation, flush=True)
                logger.append(evaluation, 'test', input_size)

            for name, module in model.named_modules():
                for attr_name in dir(module):
                    # Check if the attribute name contains 'mean_intersection_ratio'
                    if 'attn_sign_match_percentage' in attr_name or 'attn_l2_magnitude_ratio' in attr_name or 'attn_cosine_similarity' in attr_name\
                        or 'mlp_sign_match_percentage' in attr_name or 'mlp_l2_magnitude_ratio' in attr_name or 'mlp_cosine_similarity' in attr_name:
                        # Retrieve the attribute value
                        attr_value = getattr(module, attr_name)
                        # Print the module name and attribute name
                        # print('name', name, 'attr_name', attr_name, 'attr_value', attr_value)
                        # Append the attribute to the logger
                        logger.append({f'{name}_{attr_name}': attr_value}, 'test')
                        print('name', name, 'attr_name', attr_name)
                    if 'diff_ratio' in attr_name:
                        # Retrieve the attribute value
                        attr_value = getattr(module, attr_name)
                        
                            # Append the attribute to the logger
                        logger.append({f'{name}_{attr_name}': attr_value}, 'test')
                        print('name', name, attr_name, attr_value)
                    if 'cur_select_indices' in attr_name:
                        # Retrieve the attribute value
                        attr_value = getattr(module, attr_name)
                        # Append the attribute to the logger
                        logger.accumulate({f'{name}_{attr_name}': attr_value}, 'test')
                        # print('name', name, attr_name, attr_value)
                        
                    # Check if the attribute name contains 'mean_intersection_ratio'
                    

            if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
                batch_time = (time.time() - start_time) / (i + 1)
                exp_finished_time = datetime.timedelta(seconds=round(batch_time * (len(data_loader) - i - 1)))
                info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Experiment Finished Time: {}'.format(exp_finished_time)]}
                # print('running_info', info)

        if 'recordspeed' in cfg['prune_method']:
            cur_attn_inference_duration_list = []
            cur_mlp_inference_duration_list = []
            for name, module in model.named_modules():
                for attr_name in dir(module):
                    if 'cur_attn_inference_duration' in attr_name:
                        # Retrieve the attribute value
                        # if 'opt-13b' in cfg['model_name'] and get_layer_order(name) >= 20:
                        #     continue
                        attr_value = getattr(module, attr_name)
                        cur_attn_inference_duration_list.append(attr_value)
                        # logger.append({f'{name}_{attr_name}': attr_value}, 'test')
                        print('name', name, attr_name, attr_value)
                    if 'cur_mlp_inference_duration' in attr_name:
                        # diff gpu cannt measure the inference time correctly
                        # if 'opt-13b' in cfg['model_name'] and get_layer_order(name) >= 20:
                        #     continue
                        # Retrieve the attribute value
                        attr_value = getattr(module, attr_name)
                        cur_mlp_inference_duration_list.append(attr_value)
                        # logger.append({f'{name}_{attr_name}': attr_value}, 'test')
                        print('name', name, attr_name, attr_value)

            mean_cur_attn_inference_duration = sum(cur_attn_inference_duration_list)/len(cur_attn_inference_duration_list)
            mean_cur_mlp_inference_duration = sum(cur_mlp_inference_duration_list)/len(cur_mlp_inference_duration_list)
            logger.append({f'attn_inference_duration': mean_cur_attn_inference_duration}, 'test')
            logger.append({f'mlp_inference_duration': mean_cur_mlp_inference_duration}, 'test')
            print('mean_cur_attn_inference_duration', mean_cur_attn_inference_duration)
            print('mean_cur_mlp_inference_duration', mean_cur_mlp_inference_duration)
            print('mean_inference_duration', inference_duration/len(data_loader))
            print('inference_duration', inference_duration)

        if cfg['onlyprobe'] == False: 
            evaluation = metric.evaluate('test', 'full')
            print('\n\nevaluation_for_full', evaluation, '\n')
            logger.append(evaluation, 'test')
            # info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(cfg['epoch'], 100.), 'mean_inference_duration': sum(inference_duration_list)/len(inference_duration_list)]}
            info = {'info': [
                'Test Epoch: {}({:.0f}%)'.format(cfg['epoch'], 100.),
                'mean_inference_duration: {:.4f}'.format(sum(inference_duration_list)/len(inference_duration_list))
            ]}
            logger.append(info, 'test')
            print(logger.write('test', metric.metric_name['test']), flush=True)
        model_prof.stop_profile()

        

        torch.cuda.cudart().cudaProfilerStop()
    return inference_duration

from torch.utils.flop_counter import FlopCounterMode
from collections import defaultdict
import itertools
from model.data_utils import get_test_data

@torch.no_grad()
def eff_eval(model, tokenizer, dataset='wikitext2', original_len=512, generated_len=128,
             batch_size=1, device='gpu', max_time=600):
    model.eval()
    test_loader = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size=batch_size)

    is_cuda = device == 'gpu'
    if is_cuda:
        devices = [d for d in range(torch.cuda.device_count())]
        weight_memory = sum(p.element_size() * p.nelement() for p in model.parameters()) / (1024 ** 3)
    else:
        devices = []
        weight_memory = sum(p.element_size() * p.nelement() for p in model.parameters()) / (1024 ** 3)
    
    # -------------------------------
    # 第一部分：测量吞吐量（Throughput）
    # -------------------------------
    print("开始测量吞吐量...", flush=True)
    num_batches_to_fetch = 5 if device == 'gpu' else 2
    throughput_time = 0
    token_num = 0
    completed_batches = 0

    # 在评测开始前同步设备，清理缓存
    if is_cuda:
        for d in devices:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(d)

    start_time_total = time.perf_counter()

    for batch_idx, batch_data in enumerate(itertools.islice(test_loader, num_batches_to_fetch)):
        input_device = next(model.parameters()).device if is_cuda else torch.device('cpu')
        batch = batch_data.to(input_device)
        
        # 开始计时
        start_time = time.perf_counter()

        # 生成输出
        generation_output = model.generate(
            input_ids=batch,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            use_cache=True,
            top_k=50,
            max_length=original_len + generated_len,
            top_p=0.95,
            temperature=1,
        )

        # 同步设备，确保所有计算完成
        if is_cuda:
            for d in devices:
                torch.cuda.synchronize(d)

        # 结束计时
        end_time = time.perf_counter()

        batch_time = end_time - start_time
        throughput_time += batch_time
        token_num += batch.shape[0] * generated_len
        completed_batches += 1

        print(f"批次 {batch_idx + 1}/{num_batches_to_fetch} - 耗时: {batch_time:.4f}s - 生成 tokens 数: {batch.shape[0] * generated_len}", flush=True)

        # 检查是否超过最大时间限制
        if end_time - start_time_total > max_time:
            print(f"已达到最大时间限制 {max_time} 秒，停止评估。", flush=True)
            break

    total_time = time.perf_counter() - start_time_total

    if is_cuda:
        # 在评测结束后同步设备，获取内存占用
        for d in devices:
            torch.cuda.synchronize(d)
        current_memory = sum(torch.cuda.max_memory_allocated(d) for d in devices)
        activation_memory = (current_memory) / (1024 ** 3)
        memory_info = f"总内存占用: {current_memory / (1024 ** 3):.2f} GB\n" \
                      f"权重内存: {weight_memory:.2f} GB\n" \
                      f"激活值内存: {activation_memory - weight_memory:.2f} GB\n"
    else:
        memory_info = "在 CPU 上运行，无法获得内存测量数据。\n"

    avg_throughput = token_num / throughput_time if throughput_time > 0 else 0
    throughput_info = f"吞吐量: {avg_throughput:.2f} tokens/sec\n" \
                      f"完成批次数: {completed_batches}/{num_batches_to_fetch}\n" \
                      f"总评估时间: {total_time:.2f} 秒\n" \
                      f"每批次平均时间: {(throughput_time / completed_batches):.2f} 秒\n" \
                      f"生成长度: {generated_len}\n"

    # -------------------------------
    # 第二部分：测量 FLOPs
    # -------------------------------
    print("开始测量 FLOPs...", flush=True)
    total_flops = 0
    flops_completed_batches = 0
    num_batches_to_fetch_flops = 5  # 为了节省时间，FLOPs 测量可使用更少的批次

    # 重新加载数据
    test_loader_flops = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size=batch_size)

    # 同步设备
    if is_cuda:
        for d in devices:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(d)

    start_time_total_flops = time.perf_counter()

    for batch_idx, batch_data in enumerate(itertools.islice(test_loader_flops, num_batches_to_fetch_flops)):
        input_device = next(model.parameters()).device if is_cuda else torch.device('cpu')
        batch = batch_data.to(input_device)

        # 初始化 FLOPs 计数器
        flop_counter = FlopCounterMode(model, display=False)

        # Reset FLOPs count
        flop_counter.flop_counts = defaultdict(lambda: defaultdict(int))

        # 开始计时
        start_time = time.perf_counter()

        with flop_counter:
            generation_output = model.generate(
                input_ids=batch,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                use_cache=True,
                top_k=50,
                max_length=original_len + generated_len,
                top_p=0.95,
                temperature=1,
            )

        # 同步设备
        if is_cuda:
            for d in devices:
                torch.cuda.synchronize(d)

        # 结束计时
        end_time = time.perf_counter()

        batch_flops = flop_counter.get_total_flops()
        total_flops += batch_flops
        flops_completed_batches += 1

        batch_time = end_time - start_time
        print(f"批次 {batch_idx + 1}/{num_batches_to_fetch_flops} - FLOPs: {batch_flops:.2e} - Time: {batch_time:.4f}s", flush=True)
        # print(f"批次 {batch_idx + 1}/{num_batches_to_fetch_flops} - FLOPs: {batch_flops:.2e} - 耗时: {batch_time:.4f}s")

        # 检查是否超过最大时间限制
        if end_time - start_time_total_flops > max_time:
            print(f"已达到最大时间限制 {max_time} 秒，停止 FLOPs 评估。", flush=True)
            break

    total_time_flops = time.perf_counter() - start_time_total_flops

    avg_flops = total_flops / flops_completed_batches if flops_completed_batches > 0 else 0
    flops_info = f"平均每批次 FLOPs: {avg_flops:.2e}\n" \
                 f"完成批次数: {flops_completed_batches}/{num_batches_to_fetch_flops}\n" \
                 f"总评估时间: {total_time_flops:.2f} 秒\n" \
                 f"生成长度: {generated_len}\n"

    # 返回结果
    result = memory_info + "\n---- 吞吐量测量结果 ----\n" + throughput_info + "\n---- FLOPs 测量结果 ----\n" + flops_info

    return result

def convert_to_serializable(obj):
    """将不可序列化的对象转换为可序列化的格式"""
    if hasattr(obj, 'item'):  # 处理 numpy/torch 数值类型
        return obj.item()
    elif hasattr(obj, 'cpu'):  # 处理 torch.device
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    return obj


def run_lm_eval(model, tokenizer, batch_size=16, task_names=["openbookqa", "arc_easy", "winogrande", "hellaswag",
             "arc_challenge", "piqa", "mathqa"], output_dir=""):
    # Import the correct task loading function
    from lm_eval import tasks, evaluator
    import json
    from datetime import datetime
    import os
    import torch

    results = evaluator.simple_evaluate(
        model=model,
        tokenizer=tokenizer,
        tasks=task_names,
        batch_size=batch_size,
        device=next(model.parameters()).device,
        write_out=True,
        log_samples=True,
        verbosity="INFO",
        num_fewshot=0,
        task_manager=tasks.TaskManager(),
    )

    # Remove samples from results to reduce file size
    if 'samples' in results:
        del results['samples']

    # Custom JSON Encoder to handle torch.Tensor, torch.device, and numpy.ndarray
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            elif isinstance(obj, torch.device):
                return str(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"results_{timestamp}.json")

    # Save the results dictionary to a JSON file using the custom encoder
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4, cls=CustomEncoder)

    print(f"Results saved to {filename}")


import os
import random
import torch
import sys
from datasets import load_dataset
from torch.utils.data.dataset import Dataset

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

def get_calib_train_data(name, tokenizer, nsamples, seqlen=2048, seed=3, batch_size=1, dataset_cache_dir=None):
    import random
    random.seed(seed)
    cache_file = (
        f"cache/{name}_{nsamples}_{seqlen}_{seed}_{batch_size}.pt"
    )
    nsamples += 1 #############################
    if not os.path.exists("cache"):
        os.makedirs("cache")
    if os.path.exists(cache_file):
        traindataset = torch.load(cache_file)
        return traindataset
    if name == "c4":
        traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
        tot_text = "\n\n".join(traindata["text"])
    elif name == "ptb":
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
        tot_text = "\n\n".join(traindata["sentence"])
    elif name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=dataset_cache_dir)
        tot_text = "\n\n".join(traindata["text"])
    else:
        raise NotImplementedError
    traindataset = []
    for s in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            s = s - 1
            continue
        if s % batch_size == 0:
            if s != 0:
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
            inp = trainenc.input_ids[:, :seqlen]
        else:
            inp = torch.cat((inp, trainenc.input_ids[:, :seqlen]), dim=0)
    torch.save(traindataset, cache_file)
    return traindataset



# def get_wikitext2(script_args, nsamples, seqlen, tokenizer, model=None):
#     traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
#     tot_text = "\n\n".join(traindata["text"])
#     traindataset = []
    
#     # 使用滑动窗口而非随机采样
#     stride = len(tot_text) // (nsamples * 2)  # 确保有足够的样本
    
#     for i in tqdm(range(0, len(tot_text) - seqlen, stride)[:nsamples], desc="Generating WikiText2 samples"):
#         txt = tot_text[i:i+seqlen]
#         # 移除随机选择句子的逻辑
#         # 使用固定的序列长度
#         trainenc = tokenizer(txt, truncation=True, padding='max_length', max_length=seqlen, return_tensors="pt")
        
#         if trainenc.input_ids.shape[1] >= seqlen // 2:
#             sample = {
#                 "input_ids": trainenc.input_ids[0, :seqlen],
#                 "attention_mask": trainenc.attention_mask[0, :seqlen]
#             }
            
#             if script_args.use_improved_lora and model is not None:
#                 try:
#                     with torch.no_grad():
#                         outputs = model(
#                             input_ids=trainenc.input_ids.to(model.device),
#                             attention_mask=trainenc.attention_mask.to(model.device)
#                         )
#                         # 确保 dense_logits 是浮点类型并且其 vocab_size 与 tokenizer 一致
#                         if outputs.logits.shape[-1] != tokenizer.vocab_size:
#                             raise ValueError(f"Model logits vocab_size {outputs.logits.shape[-1]} does not match tokenizer vocab_size {tokenizer.vocab_size}.")
#                         sample["dense_logits"] = outputs.logits[0, :seqlen].to(torch.float32).cpu()
#                         print("Successfully generated dense_logits for sample")
#                 except Exception as e:
#                     print(f"Error generating dense_logits: {e}")
#                     script_args.use_improved_lora = False
            
#             traindataset.append(sample)
    
#     # 验证数据集
#     if script_args.use_improved_lora:
#         has_dense_logits = all('dense_logits' in item for item in traindataset)
#         print(f"Dataset validation: {len(traindataset)} samples, all have dense_logits: {has_dense_logits}")
#         if not has_dense_logits:
#             raise ValueError("Not all samples have dense_logits.")
    
#     random.shuffle(traindataset)
#     return traindataset

def get_wikitext2(script_args, nsamples, seqlen, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    tot_text = "\n\n".join(traindata["text"])
    traindataset = []
    
    # 使用更大的上下文窗口
    context_window = seqlen * 20  # 增加上下文窗口大小
    
    for _ in tqdm(range(nsamples), desc="Generating WikiText2 samples"):
        # 随机选择起始位置
        i = random.randint(0, max(0, len(tot_text) - context_window - 1))
        txt = tot_text[i:i+context_window]
        
        # 随机选择句子起始点
        sentences = txt.split('.')
        if len(sentences) > 1:
            start_sentence = random.randint(0, len(sentences) - 1)
            txt = '.'.join(sentences[start_sentence:])
        
        # 动态调整序列长度
        actual_seqlen = random.randint(seqlen // 2, seqlen)
        
        trainenc = tokenizer(txt, truncation=True, padding='max_length', max_length=actual_seqlen, return_tensors="pt")
        if trainenc.input_ids.shape[1] >= actual_seqlen // 2:  # 允许更短的序列，增加多样性
            traindataset.append({
                "input_ids": trainenc.input_ids[:, :actual_seqlen],
                "attention_mask": trainenc.attention_mask[:, :actual_seqlen]
            })
    
    # 打乱数据集
    random.shuffle(traindataset)
    return traindataset

def get_dolly(script_args, nsamples, seqlen, tokenizer):
    traindata = load_dataset("databricks/databricks-dolly-15k", split='train')
    traindataset = []
    
    # 确保 nsamples 不超过数据集大小
    nsamples = min(nsamples, len(traindata))
    
    for i in tqdm(range(nsamples), desc="Processing Dolly samples"):
        sample = traindata[i]
        txt = []
        
        # 使用 get 方法来安全地获取字段，并添加适当的前缀
        if "instruction" in sample:
            txt.append(f"Instruction: {sample.get('instruction', '')}")
        if 'context' in sample:
            txt.append(f"Context: {sample.get('context', '')}")
        if 'response' in sample:
            txt.append(f"Response: {sample.get('response', '')}")
        
        # 用换行符连接不同部分
        full_text = "\n\n".join(txt)
        
        # Tokenize
        tokenized = tokenizer(full_text, truncation=True, padding='max_length', max_length=seqlen, return_tensors="pt")
        
        if tokenized.input_ids.shape[1] >= seqlen // 2:  # 保持原有的长度检查逻辑
            traindataset.append({
                "input_ids": tokenized.input_ids.squeeze(0),
                "attention_mask": tokenized.attention_mask.squeeze(0)
            })
    
    return traindataset

def print_lora_layers(model):
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            print(f"LoRA applied to layer: {name}")
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    adapter_name_or_path: Optional[str] = field(default=None)
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset_split: str = field(default="train[:100000]", metadata={"help": "(`['train', 'test', 'eval']`):"})
    dataset_field: List[str] = field(default=None, metadata={"help": "Fields of dataset input and output."})
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    lora_r: int = field(default=64, metadata={"help": "The rank of the adapter. When passing `None` and `adapter_name_or_path` is also `None`, full fine-tuning is used."})
    init_lora_weights: Literal[True, "pissa",'loftq','gaussian'] = field(default=True, metadata={"help": ("Passing True (default) results in the LoRA initialization. Passing `pissa` results in PiSSA initialization.")})
    cache_file: str = field(default=None)
    nsamples: int = field(default=None, metadata={"help": "Number of training samples to use"})
    dataset_name: str = field(default="wikitext", metadata={"help": "Name of the dataset to use. Options: 'wikitext' or 'dolly'"})
    use_improved_lora: bool = field(default=True, metadata={"help": "Whether to use improved LoRA or standard LoRA."})
    sample_fraction: float = field(default=1, metadata={"help": "Fraction of samples to use for training"})
    learning_rate: float = field(default=3e-5, metadata={"help": "Initial learning rate for AdamW optimizer."})
    lr_scheduler_type: str = field(default="linear", metadata={"help": "Type of learning rate scheduler."})
    warmup_steps: int = field(default=500, metadata={"help": "Number of warmup steps for learning rate scheduler."})
    logging_steps: int = field(default=100, metadata={"help": "Log metrics every X updates steps."})  # 日志打印步数

@dataclass
class ScriptArguments(TrainingArguments):
    model_name_or_path: str = None
    dataset_name: str = None
    dataset_split: str = "train"
    dataset_field: str = "query response"
    nsamples: int = 5000
    cache_file: str = None
    init_lora_weights: str = "gaussian"
    lora_r: int = 64
    use_improved_lora: bool = False
    adapter_name_or_path: str = None


def train(model, tokenizer, script_args):

    model.train()

    import wandb
    wandb.init(
        project="mixtral-moe-training",  # 项目名称
        name=script_args.output_dir.split('/')[-1],  # 实验名称
        config={
            "learning_rate": script_args.learning_rate,
            "scheduler": script_args.lr_scheduler_type,
            "batch_size": script_args.per_device_train_batch_size,
            "lora_r": script_args.lora_r,
            "nsamples": script_args.nsamples,
            "model_name": script_args.model_name_or_path,
        }
    )

    # 输出当前的学习率和调度策略
    print(f"Using learning rate: {script_args.learning_rate}")
    print(f"Using learning rate scheduler: {script_args.lr_scheduler_type}")
    
    # 确认 tokenizer 的 vocab_size
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # 获取训练数据
    train_data = get_wikitext2(script_args, script_args.nsamples, 2048, tokenizer)
    # train_data = get_dolly(script_args, script_args.nsamples, 2048, tokenizer)

    class WikiTextDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]
    
    train_data = WikiTextDataset(train_data)
    

    def custom_collate_fn(batch):
        # 提取输入ID和注意力掩码
        input_ids = []
        attention_mask = []
        
        # 找出最大序列长度
        max_length = max(item['input_ids'].size(0) for item in batch)
        
        for item in batch:
            # 确保所有序列都是一维的
            if len(item['input_ids'].shape) > 1:
                item_input_ids = item['input_ids'].squeeze(0)
                item_attention_mask = item['attention_mask'].squeeze(0)
            else:
                item_input_ids = item['input_ids']
                item_attention_mask = item['attention_mask']
            
            # 如果序列长度不一致，进行填充或截断
            if item_input_ids.size(0) < max_length:
                # 填充
                pad_length = max_length - item_input_ids.size(0)
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                
                item_input_ids = torch.cat([
                    item_input_ids, 
                    torch.full((pad_length,), pad_id, dtype=item_input_ids.dtype)
                ])
                
                item_attention_mask = torch.cat([
                    item_attention_mask, 
                    torch.zeros(pad_length, dtype=item_attention_mask.dtype)
                ])
            elif item_input_ids.size(0) > max_length:
                # 截断
                item_input_ids = item_input_ids[:max_length]
                item_attention_mask = item_attention_mask[:max_length]
            
            input_ids.append(item_input_ids)
            attention_mask.append(item_attention_mask)
        
        # 堆叠为批次
        input_ids_batch = torch.stack(input_ids)
        attention_mask_batch = torch.stack(attention_mask)
        
        # 检查是否使用improved_lora
        if script_args.use_improved_lora and 'dense_logits' in batch[0]:
            dense_logits = []
            for item in batch:
                if len(item['dense_logits'].shape) > 2:
                    item_dense_logits = item['dense_logits'].squeeze(0)
                else:
                    item_dense_logits = item['dense_logits']
                
                # 确保dense_logits与input_ids长度一致
                if item_dense_logits.size(0) < max_length:
                    pad_length = max_length - item_dense_logits.size(0)
                    vocab_size = item_dense_logits.size(1)
                    item_dense_logits = torch.cat([
                        item_dense_logits,
                        torch.zeros((pad_length, vocab_size), dtype=item_dense_logits.dtype)
                    ])
                elif item_dense_logits.size(0) > max_length:
                    item_dense_logits = item_dense_logits[:max_length]
                
                dense_logits.append(item_dense_logits)
            
            dense_logits_batch = torch.stack(dense_logits)
            return {
                'input_ids': input_ids_batch,
                'attention_mask': attention_mask_batch,
                'dense_logits': dense_logits_batch
            }
        else:
            return {
                'input_ids': input_ids_batch,
                'attention_mask': attention_mask_batch
            }
    
    target_modules = ["q_proj", "o_proj", "k_proj", "v_proj",
                      "w1","w2","w3",
                      "delta_u1","delta_u2","delta_u3",
                      "delta_v1","delta_v2","delta_v3",
                      "experts_delta_v1_shared","experts_delta_v2_shared","experts_delta_v3_shared"]  

    # "block_sparse_moe.Wmean1", "block_sparse_moe.Wmean2", "block_sparse_moe.Wmean3",

    # target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "w1","w2","w3",
    #                   "delta_u1","delta_u2","delta_u3"]  

    if script_args.adapter_name_or_path is not None:
        print(f"Load {script_args.init_lora_weights} from {script_args.adapter_name_or_path}")
        model = PeftModel.from_pretrained(
            model,
            is_trainable=True
        )
    elif script_args.lora_r is not None:
        print(f"Initialized {script_args.init_lora_weights} layers")
        lora_config = LoraConfig(
            r=script_args.lora_r,
            init_lora_weights=script_args.init_lora_weights,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        
    else:
        print("Full Parameter Fine-Tuning")


    for name, param in model.named_parameters():
        if any(module in name for module in target_modules):
            param.requires_grad = True
        else:
            param.requires_grad = False

    for batch in train_data:
        for k,v in batch.items():
            batch[k]=v.to("cpu")
    data_module = dict(train_dataset=train_data, data_collator=custom_collate_fn)

    # Add CustomTrainer class definition
    from transformers import Trainer
    import torch.nn.functional as F
    
    class CustomTrainer(Trainer):
        def __init__(self, *args, script_args=None, k=0.3, lambda1=1e-4, lambda2=3e-4, lambda3=1e-4, pre_trained_model=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.k = k
            self.lambda1 = lambda1
            self.lambda2 = lambda2
            self.lambda3 = lambda3
            self.use_improved_lora = script_args.use_improved_lora if script_args else False
            self.pre_trained_model = pre_trained_model
            
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs["input_ids"].clone()
            labels[:, :-1] = inputs["input_ids"][:, 1:]
            labels[:, -1] = -100
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
            # Consistency Regularization
            if hasattr(self, 'use_improved_lora') and self.use_improved_lora and 'dense_logits' in inputs:
                # Convert dense_logits to float type
                dense_logits = inputs["dense_logits"].to(torch.float32)
                pre_trained_probs = F.softmax(dense_logits, dim=-1)
                fine_tuned_probs = F.softmax(outputs.logits, dim=-1)
                
                consistency_loss = F.kl_div(fine_tuned_probs.log(), pre_trained_probs, reduction='batchmean')
            else:
                consistency_loss = 0
                
            # Overall Loss
            total_loss = loss + self.lambda1 * consistency_loss
            
            return (total_loss, outputs) if return_outputs else total_loss
            
    class WandbCustomTrainer(CustomTrainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs["input_ids"].clone()
            labels[:, :-1] = inputs["input_ids"][:, 1:]
            labels[:, -1] = -100
            total_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            
            # 记录各种损失组件
            if self.use_improved_lora:
                wandb.log({
                    "total_loss": total_loss.item(),
                    "lm_loss": outputs["loss"].item() if isinstance(outputs, dict) else outputs[0].item(),
                    "consistency_loss": self.lambda1 * outputs.get("consistency_loss", 0),
                    # "diversity_loss": self.lambda2 * outputs.get("diversity_loss", 0),
                    # "svd_loss": self.lambda3 * outputs.get("svd_loss", 0),
                }, step=self.state.global_step)
            else:
                wandb.log({
                    "total_loss": total_loss.item(),
                    "lm_loss": outputs["loss"].item() if isinstance(outputs, dict) else outputs[0].item(),
                }, step=self.state.global_step)

            return (total_loss, outputs) if return_outputs else total_loss

        def training_step(self, model, inputs):
            # 确保训练时cache关闭
            model.config.use_cache = False
            loss = super().training_step(model, inputs)
            
            # 记录学习率
            if self.lr_scheduler is not None:
                wandb.log({
                    "learning_rate": self.lr_scheduler.get_last_lr()[0]
                }, step=self.state.global_step)
                
            return loss

    trainer = WandbCustomTrainer(
        model=model,
        script_args=script_args,
        tokenizer=tokenizer,
        args=script_args,
        **data_module
    )
    model.config.use_cache = True

    trainer.train()

    wandb.finish()
    return model, tokenizer