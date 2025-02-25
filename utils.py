import torch
import time
from dataset import collate
from config import cfg
from module import model_forward, update_model_prof, to_device
from datetime import datetime

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
    import numpy as np

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



def get_wikitext2(nsamples, seed, seqlen, tokenizer, dataset_cache_dir=None):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir=dataset_cache_dir)
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir=dataset_cache_dir)

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, tokenizer, dataset_cache_dir=None):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation', cache_dir=dataset_cache_dir)

    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
    valdata = load_dataset("json", data_files="utils/c4-validation.json")['train']

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 



def get_ptb_new(nsamples, seed, seqlen, tokenizer, dataset_cache_dir=None):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test', cache_dir=dataset_cache_dir)

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
    valdata = load_dataset("json", data_files="utils/c4-validation.json")['train']

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, tokenizer)
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, tokenizer)
        return get_c4(nsamples, seed, seqlen, tokenizer)
    
    
    
def get_test_data(name, tokenizer, seq_len=2048, batch_size = 4):
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)
    ####
    def process_data(samples, tokenizer, seq_len, field_name):
        # 处理流式数据集
        if hasattr(samples, '_ex_iterable'):  # 检查是否为流式数据集
            all_text = []
            for sample in samples:
                all_text.append(sample[field_name])
            text = "\n\n".join(all_text)
        elif isinstance(samples, list):  # 处理C4数据集的情况
            text = "\n\n".join(sample[field_name] for sample in samples)
        else:
            text = "\n\n".join(samples[field_name])
            
        test_ids = tokenizer(text, return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)
    ####
    if 'wikitext2' in name:
        test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', trust_remote_code=True)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'ptb' in name:
        test_data = load_dataset('ptb_text_only', 'penn_treebank', split='test', trust_remote_code=True)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    elif 'c4' in name:
        traindata = load_dataset(
            'allenai/c4', 
            'en', 
            data_files={'train': ['en/c4-train.00000-of-01024.json.gz']},
            streaming=True, trust_remote_code=True  # 使用流式加载来处理大数据集
        )['train']
        
        # Convert streaming dataset to list and randomly sample 5000 items
        all_data = []
        for item in traindata:
            all_data.append(item)
            if len(all_data) >= 10000:  # Collect slightly more than needed to ensure enough valid samples
                break
                
        random.seed(42)  # Set seed for reproducibility
        traindata = random.sample(all_data, 5000)
        test_dataset = process_data(traindata, tokenizer, seq_len, 'text')
        # trainloader = []
        # # 遍历整个数据集
        # for i in range(len(traindata)):
        #     trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
        #     # 如果文本长度足够，进行处理
        #     if trainenc.input_ids.shape[1] >= seq_len:
        #         # 对于较长的文本，可以滑动窗口获取多个序列
        #         for start_idx in range(0, trainenc.input_ids.shape[1] - seq_len, seq_len):
        #             inp = trainenc.input_ids[:, start_idx:start_idx + seq_len]
        #             tar = inp.clone()
        #             tar[:, :-1] = -100
        #             trainloader.append((inp, tar))

        # return trainloader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

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



import numpy as np
from tqdm import tqdm

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