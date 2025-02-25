import json

def calculate_mixtral_params(config):
    """计算 Mixtral-8x7B 模型参数量(单位:B)"""
    # 设置默认值
    hidden_size = config.get("hidden_size", 4096)
    num_layers = config.get("num_hidden_layers", 32)
    vocab_size = config.get("vocab_size", 32000)
    intermediate_size = config.get("intermediate_size", 14336)
    num_experts = config.get("num_local_experts", 8)
    num_active_experts = config.get("num_experts_per_tok", 2)

    # 基础组件参数计算
    attention_intermediate_size = hidden_size // 4
    
    # 单个Expert参数
    single_expert_params = 3 * hidden_size * intermediate_size  # w1, w2, w3
    
    # 单个Router参数
    single_router_params = hidden_size * num_experts
    
    # 单个Attention参数
    single_attention_params = 2 * (hidden_size * hidden_size + hidden_size * attention_intermediate_size)
    
    # 单个Decoder Block参数
    single_decoder_params = (
        num_experts * single_expert_params +  # 8个expert
        single_router_params +  # router
        single_attention_params +  # attention
        2 * hidden_size  # 2个layernorm
    )
    
    # 推理时的单个Decoder Block参数
    inference_decoder_params = (
        num_active_experts * single_expert_params +  # 2个expert
        single_router_params +  # router
        single_attention_params +  # attention
        2 * hidden_size  # 2个layernorm
    )
    
    # 总参数量计算
    embedding_params = vocab_size * hidden_size
    transformer_params = single_decoder_params * num_layers
    inference_transformer_params = inference_decoder_params * num_layers
    output_params = hidden_size * vocab_size + hidden_size
    
    total_params = embedding_params + transformer_params + output_params
    inference_total_params = embedding_params + inference_transformer_params + output_params
    
    # 转换为 Billion
    total_params_b = total_params / (1000 ** 3)
    inference_total_params_b = inference_total_params / (1000 ** 3)
    
    details = {
        "单个Expert参数 (B)": single_expert_params / (1000 ** 3),
        "单个Router参数 (B)": single_router_params / (1000 ** 3),
        "单个Attention参数 (B)": single_attention_params / (1000 ** 3),
        "单个Decoder Block参数 (B)": single_decoder_params / (1000 ** 3),
        "Embedding层参数 (B)": embedding_params / (1000 ** 3),
        "所有Transformer层参数 (B)": transformer_params / (1000 ** 3),
        "输出层参数 (B)": output_params / (1000 ** 3),
        "总参数量 (B)": total_params_b,
        "推理时总参数量 (B)": inference_total_params_b
    }
    
    return details, total_params_b, inference_total_params_b


def calculate_compressed_mixtral_params(
    config,
    delta_ratio=0.5,
    share_V=False,
    use_pp=False,
    pp_ratio=0.2,
    ignore_layers = []
):
    # 设置默认值
    hidden_size = config.get("hidden_size", 4096)
    num_layers = config.get("num_hidden_layers", 32)
    vocab_size = config.get("vocab_size", 32000)
    intermediate_size = config.get("intermediate_size", 14336)
    num_experts = config.get("num_local_experts", 8)
    num_active_experts = config.get("num_experts_per_tok", 2)


    # Embedding 层参数
    embedding_params = vocab_size * hidden_size

    delta_low_rank = int(hidden_size * intermediate_size * delta_ratio / (hidden_size + intermediate_size))

    single_attention_params = (hidden_size * hidden_size +  # q_proj
        hidden_size * (hidden_size // 4) * 2 +  # k_proj, v_proj (使用 grouped query attention)
        hidden_size * hidden_size  # o_proj
    )

    single_router_gate_params = hidden_size * num_experts
    
    not_compressed_per_layer_params = (
        # 自注意力参数
        single_attention_params +
        # Router 参数
        single_router_gate_params +
        # MoE 参数
        num_experts * (
            hidden_size * intermediate_size * 2 +  # w1, w3
            intermediate_size * hidden_size  # w2
        ) +
        # LayerNorm 参数
        hidden_size * 2  # input_layernorm, post_attention_layernorm
    )

    inference_not_compressed_per_layer_params = (
        single_attention_params +
        single_router_gate_params +
        num_active_experts * (
            hidden_size * intermediate_size * 2 +  # Wmean1, Wmean3
            intermediate_size * hidden_size  # Wmean2
        ) +
        hidden_size * 2  # input_layernorm, post_attention_layernorm
    )
    
    if share_V == False and use_pp == False:
        # 每个 Transformer 层参数
        per_layer_params = (
            # 自注意力参数
            single_attention_params +
            # Router 参数
            single_router_gate_params +
            # MoE 参数
            (
                hidden_size * intermediate_size * 2 +  # Wmean1, Wmean3
                intermediate_size * hidden_size  # Wmean2
            ) +
            num_experts * (
                delta_low_rank * intermediate_size +  # delta_u1
                hidden_size * delta_low_rank + # delta_v1
                delta_low_rank * hidden_size + # delta_u2
                intermediate_size * delta_low_rank + # delta_v2
                delta_low_rank * intermediate_size +  # delta_u3
                hidden_size * delta_low_rank # delta_v3
            ) +
            
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )
        inference_per_layer_params = (
            # 自注意力参数
            single_attention_params +
            # Router 参数
            single_router_gate_params +
            # MoE 参数
            num_active_experts * (
                hidden_size * intermediate_size * 2 +  # Wmean1, Wmean3
                intermediate_size * hidden_size  # Wmean2
            ) +
            num_active_experts * (
                delta_low_rank * intermediate_size +  # delta_u1
                hidden_size * delta_low_rank + # delta_v1
                delta_low_rank * hidden_size + # delta_u2
                intermediate_size * delta_low_rank + # delta_v2
                delta_low_rank * intermediate_size +  # delta_u3
                hidden_size * delta_low_rank # delta_v3
            ) +
            
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )

        inference_per_layer_params_share = (
            # 自注意力参数
            single_attention_params +
            # Router 参数
            single_router_gate_params +
            # MoE 参数
            (
                hidden_size * intermediate_size * 2  # Wmean1, Wmean3
            ) +
            num_active_experts * (
                intermediate_size * hidden_size + # Wmean2
                delta_low_rank * intermediate_size +  # delta_u1
                hidden_size * delta_low_rank + # delta_v1
                delta_low_rank * hidden_size + # delta_u2
                intermediate_size * delta_low_rank + # delta_v2
                delta_low_rank * intermediate_size +  # delta_u3
                hidden_size * delta_low_rank # delta_v3
            ) +
            
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )
    elif share_V == False and use_pp == True:
    # 每个 Transformer 层参数
        per_layer_params = (
            # 自注意力参数
            single_attention_params +
            # Router 参数
            single_router_gate_params +
            # MoE 参数
            (
                hidden_size * intermediate_size * 2 +  # Wmean1, Wmean3
                intermediate_size * hidden_size  # Wmean2
            ) * (1 - pp_ratio) +
            num_experts * (
                delta_low_rank * intermediate_size +  # delta_u1
                hidden_size * delta_low_rank + # delta_v1
                delta_low_rank * hidden_size + # delta_u2
                intermediate_size * delta_low_rank + # delta_v2
                delta_low_rank * intermediate_size +  # delta_u3
                hidden_size * delta_low_rank # delta_v3
            ) +
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )
        inference_per_layer_params = (
            # 自注意力参数
            single_attention_params +
            # Router 参数
            single_router_gate_params +
            # MoE 参数
            num_active_experts * (
                hidden_size * intermediate_size * 2 +  # Wmean1, Wmean3
                intermediate_size * hidden_size  # Wmean2
            ) * (1 - pp_ratio) +
            num_active_experts * (
                delta_low_rank * intermediate_size +  # delta_u1
                hidden_size * delta_low_rank + # delta_v1
                delta_low_rank * hidden_size + # delta_u2
                intermediate_size * delta_low_rank + # delta_v2
                delta_low_rank * intermediate_size +  # delta_u3
                hidden_size * delta_low_rank # delta_v3
            ) +
            
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )

        inference_per_layer_params_share = (
            # 自注意力参数
            single_attention_params +
            # Router 参数
            single_router_gate_params +
            # MoE 参数
            (
                hidden_size * intermediate_size * 2  # Wmean1, Wmean3
            ) * (1 - pp_ratio) +
            num_active_experts * (
                intermediate_size * hidden_size * (1 - pp_ratio) + # Wmean2
                delta_low_rank * intermediate_size +  # delta_u1
                hidden_size * delta_low_rank + # delta_v1
                delta_low_rank * hidden_size + # delta_u2
                intermediate_size * delta_low_rank + # delta_v2
                delta_low_rank * intermediate_size +  # delta_u3
                hidden_size * delta_low_rank # delta_v3
            ) +
            
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )
    elif share_V == True and use_pp == False:
        per_layer_params = (
            # 自注意力参数
            single_attention_params +
            # Router 参数
            single_router_gate_params +
            # MoE 参数
            (
                hidden_size * intermediate_size * 2 +  # Wmean1, Wmean3
                intermediate_size * hidden_size  # Wmean2
            ) +
            (
                hidden_size * delta_low_rank + # shared delta_v1
                intermediate_size * delta_low_rank + # shared delta_v2
                hidden_size * delta_low_rank # shared delta_v3
            ) +
            num_experts * (
                delta_low_rank * intermediate_size +  # delta_u1
                delta_low_rank * hidden_size + # delta_u2
                delta_low_rank * intermediate_size  # delta_u3
            ) +
            
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )
        inference_per_layer_params = (
            # 自注意力参数
            single_attention_params +
            # Router 参数
            single_router_gate_params +
            # MoE 参数
            num_active_experts * (
                hidden_size * intermediate_size * 2 +  # Wmean1, Wmean3
                intermediate_size * hidden_size  # Wmean2
            ) +
            num_active_experts * (
                hidden_size * delta_low_rank + # shared delta_v1
                intermediate_size * delta_low_rank + # shared delta_v2
                hidden_size * delta_low_rank # shared delta_v3
            ) +
            num_active_experts * (
                delta_low_rank * intermediate_size +  # delta_u1
                delta_low_rank * hidden_size + # delta_u2
                delta_low_rank * intermediate_size  # delta_u3
            ) +
            
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )
        inference_per_layer_params_share = (
            # 自注意力参数
            single_attention_params +
            # Router 参数
            single_router_gate_params +
            # MoE 参数
            (
                hidden_size * intermediate_size * 2  # Wmean1, Wmean3
            ) +
            (
                hidden_size * delta_low_rank + # shared delta_v1
                hidden_size * delta_low_rank # shared delta_v3
            ) +
            num_active_experts * (
                intermediate_size * hidden_size + # Wmean2
                intermediate_size * delta_low_rank + # shared delta_v2
                delta_low_rank * intermediate_size +  # delta_u1
                delta_low_rank * hidden_size + # delta_u2
                delta_low_rank * intermediate_size  # delta_u3
            ) +
            
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )
    elif share_V == True and use_pp == True:
        per_layer_params = (
            # 自注意力参数
            single_attention_params +
            # Router 参数
            single_router_gate_params +
            # MoE 参数
            (
                hidden_size * intermediate_size * 2 +  # Wmean1, Wmean3
                intermediate_size * hidden_size  # Wmean2
            ) * (1 - pp_ratio) +
            (
                hidden_size * delta_low_rank + # shared delta_v1
                intermediate_size * delta_low_rank + # shared delta_v2
                hidden_size * delta_low_rank # shared delta_v3
            ) +
            num_experts * (
                delta_low_rank * intermediate_size +  # delta_u1
                delta_low_rank * hidden_size + # delta_u2
                delta_low_rank * intermediate_size  # delta_u3
            ) +
            
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )
        inference_per_layer_params = (
            # 自注意力参数
            single_attention_params +
            # Router 参数
            single_router_gate_params +
            # MoE 参数
            num_active_experts * (
                hidden_size * intermediate_size * 2 +  # Wmean1, Wmean3
                intermediate_size * hidden_size  # Wmean2
            ) * (1 - pp_ratio) +
            num_active_experts * (
                hidden_size * delta_low_rank + # shared delta_v1
                intermediate_size * delta_low_rank + # shared delta_v2
                hidden_size * delta_low_rank # shared delta_v3
            ) +
            num_active_experts * (
                delta_low_rank * intermediate_size +  # delta_u1
                delta_low_rank * hidden_size + # delta_u2
                delta_low_rank * intermediate_size  # delta_u3
            ) +
            
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )
        inference_per_layer_params_share = (
            # 自注意力参数
            single_attention_params +
            # Router 参数
            single_router_gate_params +
            # MoE 参数
            (
                hidden_size * intermediate_size * 2   # Wmean1, Wmean3
            ) * (1 - pp_ratio) +
            (
                hidden_size * delta_low_rank + # shared delta_v1
                hidden_size * delta_low_rank # shared delta_v3
            ) +
            num_active_experts * (
                intermediate_size * hidden_size + # Wmean2
                intermediate_size * delta_low_rank + # shared delta_v2
                delta_low_rank * intermediate_size +  # delta_u1
                delta_low_rank * hidden_size + # delta_u2
                delta_low_rank * intermediate_size  # delta_u3
            ) +
            
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )
    
    # 所有 Transformer 层的参数
    transformer_params = per_layer_params * (num_layers - len(ignore_layers)) + len(ignore_layers) * not_compressed_per_layer_params
    inference_transformer_params = inference_per_layer_params * (num_layers - len(ignore_layers)) + len(ignore_layers) * inference_not_compressed_per_layer_params
    inference_transformer_params_share = inference_per_layer_params_share * (num_layers - len(ignore_layers)) + len(ignore_layers) * inference_not_compressed_per_layer_params
    # 输出层参数
    output_params = hidden_size * vocab_size + hidden_size  # lm_head + final_norm
    
    # 总参数量
    total_params = embedding_params + transformer_params + output_params
    inference_total_params = embedding_params + inference_transformer_params + output_params
    inference_total_params_share = embedding_params + inference_transformer_params_share + output_params
    # 转换为 Billion
    total_params_b = total_params / (1000 ** 3)
    inference_total_params_b = inference_total_params / (1000 ** 3)
    inference_total_params_share_b = inference_total_params_share / (1000 ** 3)
    # 详细分布
    details = {
        "Embedding 层 (B)": embedding_params / (1000 ** 3),
        "每个 Transformer 层 (B)": per_layer_params / (1000 ** 3),
        "所有 Transformer 层 (B)": transformer_params / (1000 ** 3),
        "输出层 (B)": output_params / (1000 ** 3),
        "总参数量 (B)": total_params_b,
        "推理时总参数量 (B)": inference_total_params_b,
        "推理时共享总参数量 (B)": inference_total_params_share_b
    }
    
    return details, total_params_b, inference_total_params_b, inference_total_params_share_b


def find_delta_ratio(target_compression_ratio=0.2, tolerance=1e-4, use_pp=False, share_V=False, pp_ratio=0.2, ignore_layers = [],
                     config=None):
    """
    通过二分查找找到合适的delta_ratio值
    Args:
        target_compression_ratio: 目标压缩比例，如0.2表示压缩20%
        tolerance: 允许的误差范围
    Returns:
        找到的delta_ratio值
    """
    # 获取原始模型参数量
    _, original_params, _ = calculate_mixtral_params(config=config)
    target_params = original_params * (1 - target_compression_ratio)
    
    # 二分查找的范围
    left, right = 0.0, 1.0
    
    while right - left > tolerance:
        mid = (left + right) / 2
        _, compressed_params, _ , _ = calculate_compressed_mixtral_params(delta_ratio=mid, use_pp=use_pp, share_V=share_V, pp_ratio=pp_ratio, ignore_layers = ignore_layers, config=config)
        
        if compressed_params > target_params:
            # 如果当前参数量太大，需要更小的delta_ratio
            right = mid
        else:
            # 如果当前参数量太小，需要更大的delta_ratio
            left = mid
    
    final_ratio = (left + right) / 2
    _, final_params, _ , _ = calculate_compressed_mixtral_params(delta_ratio=final_ratio, use_pp=use_pp, share_V=share_V, pp_ratio=pp_ratio, ignore_layers=ignore_layers, config=config)
    
    print(f"Original params: {original_params:.2f}B")
    print(f"Target params: {target_params:.2f}B")
    print(f"Achieved params: {final_params:.2f}B")
    print(f"Compression ratio: {(original_params - final_params) / original_params:.2%}")
    
    return final_ratio

def calculate_mixtral_memory(
    batch_size=1,
    sequence_length=2048,
    hidden_size=4096,
    num_layers=32,
    vocab_size=32000,
    num_attention_heads=32,
    num_key_value_heads=8,
    intermediate_size=14336,
    num_experts=8,
    dtype_size=2,  # fp16 = 2 bytes
    training=False
):
    """
    计算 Mixtral-8x7B 模型显存占用
    """
    # 1. 模型参数显存
    # Embedding 层
    embedding_param = vocab_size * hidden_size
    
    # 每个 Transformer 层参数
    per_layer_params = (
        # 自注意力参数
        hidden_size * hidden_size +  # q_proj
        hidden_size * (hidden_size // 4) * 2 +  # k_proj, v_proj
        hidden_size * hidden_size +  # o_proj
        # MoE 参数
        hidden_size * num_experts +  # gate
        num_experts * (
            hidden_size * intermediate_size * 2 +  # w1, w3
            intermediate_size * hidden_size  # w2
        ) +
        # LayerNorm 参数
        hidden_size * 2  # input_layernorm, post_attention_layernorm
    )
    
    transformer_params = per_layer_params * num_layers
    
    # 输出层参数
    output_params = hidden_size * vocab_size + hidden_size  # lm_head + final_norm
    
    total_params = embedding_param + transformer_params + output_params
    param_memory = total_params * dtype_size / (1024**3)  # 转换为 GB
    
    # 2. 激活值显存
    # 每个 token 的隐藏状态
    hidden_memory = batch_size * sequence_length * hidden_size * dtype_size
    
    # 注意力 key/value 缓存
    kv_cache = batch_size * sequence_length * (hidden_size // 2) * num_layers * dtype_size
    
    activation_memory = (hidden_memory + kv_cache) / (1024**3)  # 转换为 GB
    
    # 3. 如果是训练模式，需要考虑优化器状态
    optimizer_memory = 0
    if training:
        optimizer_memory = total_params * dtype_size * 2 / (1024**3)  # Adam 优化器需要 2 个状态
    
    total_memory = param_memory + activation_memory + optimizer_memory
    
    return {
        "参数显存 (GB)": param_memory,
        "激活值显存 (GB)": activation_memory,
        "优化器显存 (GB)": optimizer_memory,
        "总显存 (GB)": total_memory
    }

def calculate_deepseek_params(config):
    """计算 DeepSeek 模型参数量(单位:B)"""
    # 设置默认值
    hidden_size = config.get("hidden_size", 2048)
    num_layers = config.get("num_hidden_layers", 28)  # 28层 (0-27)
    vocab_size = config.get("vocab_size", 102400)
    first_layer_intermediate_size = config.get("first_layer_intermediate_size", 10944)
    moe_intermediate_size = config.get("moe_intermediate_size", 1408)
    shared_expert_intermediate_size = config.get("shared_expert_intermediate_size", 2816)
    num_experts = config.get("num_local_experts", 64)
    num_active_experts = config.get("num_experts_per_tok", 2)  # 推理时激活的专家数

    # 基础组件参数计算
    # Attention参数 (所有层相同)
    single_attention_params = 4 * (hidden_size * hidden_size)  # q_proj, k_proj, v_proj, o_proj
    
    # 第一层MLP参数
    first_layer_mlp_params = 3 * hidden_size * first_layer_intermediate_size  # gate_proj, up_proj, down_proj
    
    # MoE层参数计算
    single_expert_params = 3 * hidden_size * moe_intermediate_size  # gate_proj, up_proj, down_proj
    shared_expert_params = 3 * hidden_size * shared_expert_intermediate_size  # shared expert的参数
    single_router_params = hidden_size * num_experts  # gate参数
    
    # 单个MoE Decoder Block参数
    moe_decoder_params = (
        num_experts * single_expert_params +  # 64个expert
        shared_expert_params +  # 共享expert
        single_router_params +  # router
        single_attention_params +  # attention
        2 * hidden_size  # 2个layernorm
    )
    
    # 第一层Decoder Block参数
    first_decoder_params = (
        first_layer_mlp_params +  # MLP
        single_attention_params +  # attention
        2 * hidden_size  # 2个layernorm
    )
    
    # 推理时的单个MoE Decoder Block参数
    inference_moe_decoder_params = (
        num_active_experts * single_expert_params +  # 激活的expert
        shared_expert_params +  # 共享expert
        single_router_params +  # router
        single_attention_params +  # attention
        2 * hidden_size  # 2个layernorm
    )
    
    # 总参数量计算
    embedding_params = vocab_size * hidden_size  # 词嵌入
    transformer_params = first_decoder_params + moe_decoder_params * (num_layers - 1)  # 第一层 + 其他MoE层
    inference_transformer_params = first_decoder_params + inference_moe_decoder_params * (num_layers - 1)
    output_params = hidden_size * vocab_size  # lm_head
    final_norm_params = hidden_size  # 最终的norm层
    
    total_params = embedding_params + transformer_params + output_params + final_norm_params
    inference_total_params = embedding_params + inference_transformer_params + output_params + final_norm_params
    
    # 转换为 Billion
    total_params_b = total_params / (1000 ** 3)
    inference_total_params_b = inference_total_params / (1000 ** 3)
    
    details = {
        "词嵌入参数 (B)": embedding_params / (1000 ** 3),
        "第一层Decoder参数 (B)": first_decoder_params / (1000 ** 3),
        "单个Expert参数 (B)": single_expert_params / (1000 ** 3),
        "共享Expert参数 (B)": shared_expert_params / (1000 ** 3),
        "单个Router参数 (B)": single_router_params / (1000 ** 3),
        "单个Attention参数 (B)": single_attention_params / (1000 ** 3),
        "单个MoE Decoder参数 (B)": moe_decoder_params / (1000 ** 3),
        "所有Transformer层参数 (B)": transformer_params / (1000 ** 3),
        "输出层参数 (B)": output_params / (1000 ** 3),
        "总参数量 (B)": total_params_b,
        "推理时总参数量 (B)": inference_total_params_b
    }
    
    return details, total_params_b, inference_total_params_b

def calculate_qwen_moe_params(config):
    """Calculate Qwen MoE model parameters (in billions)"""
    # Default values
    hidden_size = config.get("hidden_size", 3584)
    num_layers = config.get("num_hidden_layers", 28)
    vocab_size = config.get("vocab_size", 151936)
    expert_hidden_size = config.get("expert_hidden_size", 2560)  # MLP hidden size for each expert
    shared_expert_hidden_size = config.get("shared_expert_hidden_size", 20480)  # Shared expert hidden size
    num_experts = config.get("num_local_experts", 64)
    num_active_experts = config.get("num_experts_per_tok", 2)  # Number of active experts during inference

    # Basic component parameter calculations
    # Attention parameters (same for all layers)
    single_attention_params = (
        hidden_size * hidden_size +  # q_proj
        hidden_size * (hidden_size // 7) * 2 +  # k_proj, v_proj (using 512 head dim)
        hidden_size * hidden_size  # o_proj
    )

    # MoE layer parameters
    single_expert_params = 3 * hidden_size * expert_hidden_size  # gate_proj, up_proj, down_proj
    shared_expert_params = 3 * hidden_size * shared_expert_hidden_size  # shared expert parameters
    single_router_params = hidden_size * num_experts  # gate parameters
    shared_expert_gate_params = hidden_size * 1  # shared expert gate

    # Single MoE Decoder Block parameters
    moe_decoder_params = (
        num_experts * single_expert_params +  # expert MLPs
        shared_expert_params +  # shared expert
        single_router_params +  # router
        shared_expert_gate_params +  # shared expert gate
        single_attention_params +  # attention
        2 * hidden_size  # 2 layer norms
    )

    # Inference time single MoE Decoder Block parameters
    inference_moe_decoder_params = (
        num_active_experts * single_expert_params +  # active expert MLPs
        shared_expert_params +  # shared expert
        single_router_params +  # router
        shared_expert_gate_params +  # shared expert gate
        single_attention_params +  # attention
        2 * hidden_size  # 2 layer norms
    )

    # Total parameter calculations
    embedding_params = vocab_size * hidden_size  # token embeddings
    transformer_params = moe_decoder_params * num_layers  # all transformer layers
    inference_transformer_params = inference_moe_decoder_params * num_layers
    output_params = hidden_size * vocab_size  # lm_head
    final_norm_params = hidden_size  # final norm layer

    total_params = embedding_params + transformer_params + output_params + final_norm_params
    inference_total_params = embedding_params + inference_transformer_params + output_params + final_norm_params

    # Convert to billions
    total_params_b = total_params / (1000 ** 3)
    inference_total_params_b = inference_total_params / (1000 ** 3)

    details = {
        "Embedding params (B)": embedding_params / (1000 ** 3),
        "Single Expert params (B)": single_expert_params / (1000 ** 3),
        "Shared Expert params (B)": shared_expert_params / (1000 ** 3),
        "Single Router params (B)": single_router_params / (1000 ** 3),
        "Single Attention params (B)": single_attention_params / (1000 ** 3),
        "Single MoE Decoder params (B)": moe_decoder_params / (1000 ** 3),
        "All Transformer layers params (B)": transformer_params / (1000 ** 3),
        "Output layer params (B)": output_params / (1000 ** 3),
        "Total params (B)": total_params_b,
        "Inference total params (B)": inference_total_params_b
    }

    return details, total_params_b, inference_total_params_b


def calculate_compressed_deepseek_params(
    config,
    delta_ratio=0.5,
    share_V=False,
    use_pp=False,
    pp_ratio=0.2,
):
    # 设置默认值
    hidden_size = config.get("hidden_size", 2048)
    num_layers = config.get("num_hidden_layers", 28)  # 28层 (0-27)
    vocab_size = config.get("vocab_size", 102400)
    first_layer_intermediate_size = config.get("first_layer_intermediate_size", 10944)
    moe_intermediate_size = config.get("moe_intermediate_size", 1408)
    shared_expert_intermediate_size = config.get("shared_expert_intermediate_size", 2816)
    num_experts = config.get("num_local_experts", 64)
    num_active_experts = config.get("num_experts_per_tok", 2)  # 推理时激活的专家数

    # Embedding 层参数
    embedding_params = vocab_size * hidden_size

    # Attention参数 (所有层相同)
    single_attention_params = 4 * (hidden_size * hidden_size)  # q_proj, k_proj, v_proj, o_proj
    # ------------------------------------------------------------------------------------------
    # 第一层MLP参数
    first_layer_mlp_params = 3 * hidden_size * first_layer_intermediate_size  # gate_proj, up_proj, down_proj
    
    # 第一层Decoder Block参数
    first_decoder_params = (
        first_layer_mlp_params +  # MLP
        single_attention_params +  # attention
        2 * hidden_size  # 2个layernorm
    )
    # ------------------------------------------------------------------------------------------
    delta_low_rank = int(hidden_size * moe_intermediate_size * delta_ratio / (hidden_size + moe_intermediate_size))
    
    single_router_params = hidden_size * num_experts  # gate参数  
    shared_expert_params = 3 * hidden_size * shared_expert_intermediate_size  # shared expert的参数

    if use_pp == False:
        # 每个 Transformer 层参数
        moe_decoder_params = (
            # 自注意力参数
            single_attention_params +
            # Router 参数
            single_router_params +
            # shared expert 参数
            shared_expert_params +
            # MoE 参数
            (
                hidden_size * moe_intermediate_size * 2 +  # Wmean1, Wmean3
                moe_intermediate_size * hidden_size  # Wmean2
            ) +
            num_experts * (
                delta_low_rank * moe_intermediate_size +  # delta_u1
                hidden_size * delta_low_rank + # delta_v1
                delta_low_rank * hidden_size + # delta_u2
                moe_intermediate_size * delta_low_rank + # delta_v2
                delta_low_rank * moe_intermediate_size +  # delta_u3
                hidden_size * delta_low_rank # delta_v3
            ) +
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )
        inference_moe_decoder_params = (
            # 自注意力参数
            single_attention_params +
            # Router 参数
            single_router_params +
            # shared expert 参数
            shared_expert_params +
            # MoE 参数
            (
                hidden_size * moe_intermediate_size * 2  # Wmean1, Wmean3
            ) +
            num_active_experts * (
                moe_intermediate_size * hidden_size + # Wmean2
                delta_low_rank * moe_intermediate_size +  # delta_u1
                hidden_size * delta_low_rank + # delta_v1
                delta_low_rank * hidden_size + # delta_u2
                moe_intermediate_size * delta_low_rank + # delta_v2
                delta_low_rank * moe_intermediate_size +  # delta_u3
                hidden_size * delta_low_rank # delta_v3
            ) +
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )
    elif use_pp == True:
        # 每个 Transformer 层参数
        moe_decoder_params = (
            # 自注意力参数
            single_attention_params +
            # Router 参数
            single_router_params +
            # shared expert 参数
            shared_expert_params +
            # MoE 参数
            (
                hidden_size * moe_intermediate_size * 2 +  # Wmean1, Wmean3
                moe_intermediate_size * hidden_size  # Wmean2
            ) * (1 - pp_ratio) +
            num_experts * (
                delta_low_rank * moe_intermediate_size +  # delta_u1
                hidden_size * delta_low_rank + # delta_v1
                delta_low_rank * hidden_size + # delta_u2
                moe_intermediate_size * delta_low_rank + # delta_v2
                delta_low_rank * moe_intermediate_size +  # delta_u3
                hidden_size * delta_low_rank # delta_v3
            ) +
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )
        inference_moe_decoder_params = (
            # 自注意力参数
            single_attention_params +
            # Router 参数
            single_router_params +
            # shared expert 参数
            shared_expert_params +
            # MoE 参数
            (
                hidden_size * moe_intermediate_size * 2  # Wmean1, Wmean3
            ) * (1 - pp_ratio) +
            num_active_experts * (
                moe_intermediate_size * hidden_size + # Wmean2
                delta_low_rank * moe_intermediate_size +  # delta_u1
                hidden_size * delta_low_rank + # delta_v1
                delta_low_rank * hidden_size + # delta_u2
                moe_intermediate_size * delta_low_rank + # delta_v2
                delta_low_rank * moe_intermediate_size +  # delta_u3
                hidden_size * delta_low_rank # delta_v3
            ) +
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )
    
    # 所有 Transformer 层的参数
    transformer_params = moe_decoder_params * (num_layers - 1) + first_decoder_params
    inference_transformer_params = first_decoder_params + inference_moe_decoder_params * (num_layers - 1)
    # 输出层参数
    output_params = hidden_size * vocab_size + hidden_size  # lm_head + final_norm
    
    # 总参数量
    total_params = embedding_params + transformer_params + output_params
    inference_total_params = embedding_params + inference_transformer_params + output_params
    # 转换为 Billion
    total_params_b = total_params / (1000 ** 3)
    inference_total_params_b = inference_total_params / (1000 ** 3)
    # 详细分布
    details = {
        "Embedding 层 (B)": embedding_params / (1000 ** 3),
        "所有 Transformer 层 (B)": transformer_params / (1000 ** 3),
        "输出层 (B)": output_params / (1000 ** 3),
        "总参数量 (B)": total_params_b,
        "推理时总参数量 (B)": inference_total_params_b,
    }
    
    return details, total_params_b, inference_total_params_b



def calculate_compressed_qwen_moe_params(
    config,
    delta_ratio=0.5,
    share_V=False,
    use_pp=False,
    pp_ratio=0.2,
    ignore_layers = [],
):
    # 设置默认值
    hidden_size = config.get("hidden_size", 3584)
    num_layers = config.get("num_hidden_layers", 28)
    vocab_size = config.get("vocab_size", 151936)
    expert_hidden_size = config.get("expert_hidden_size", 2560)  # MLP hidden size for each expert
    shared_expert_hidden_size = config.get("shared_expert_hidden_size", 20480)  # Shared expert hidden size
    num_experts = config.get("num_local_experts", 64)
    num_active_experts = config.get("num_experts_per_tok", 2)  # Number of active experts during inference

    # Embedding 层参数
    embedding_params = vocab_size * hidden_size

    # Attention参数 (所有层相同)
    single_attention_params = (
        hidden_size * hidden_size +  # q_proj
        hidden_size * (hidden_size // 7) * 2 +  # k_proj, v_proj (using 512 head dim)
        hidden_size * hidden_size  # o_proj
    )

    delta_low_rank = int(hidden_size * expert_hidden_size * delta_ratio / (hidden_size + expert_hidden_size))
    
    shared_expert_params = 3 * hidden_size * shared_expert_hidden_size  # shared expert parameters
    single_router_params = hidden_size * num_experts  # gate parameters
    shared_expert_gate_params = hidden_size * 1  # shared expert gate


    uncompressed_moe_decoder_params = (
        num_experts * 3 * hidden_size * expert_hidden_size +  # expert MLPs
        shared_expert_params +  # shared expert
        single_router_params +  # router
        shared_expert_gate_params +  # shared expert gate
        single_attention_params +  # attention
        2 * hidden_size  # 2 layer norms
    )

    uncompressed_inference_moe_decoder_params = (
        num_active_experts * 3 * hidden_size * expert_hidden_size +  # active expert MLPs
        shared_expert_params +  # shared expert
        single_router_params +  # router
        shared_expert_gate_params +  # shared expert gate
        single_attention_params +  # attention
        2 * hidden_size  # 2 layer norms
    )


    if use_pp == False:
        # 每个 Transformer 层参数
        moe_decoder_params = (
            # 自注意力参数
            single_attention_params +
            # Router 参数
            single_router_params +
            # shared expert 参数
            shared_expert_params +
            shared_expert_gate_params +
            # MoE 参数
            (
                hidden_size * expert_hidden_size * 2 +  # Wmean1, Wmean3
                expert_hidden_size * hidden_size  # Wmean2
            ) +
            num_experts * (
                delta_low_rank * expert_hidden_size +  # delta_u1
                hidden_size * delta_low_rank + # delta_v1
                delta_low_rank * hidden_size + # delta_u2
                expert_hidden_size * delta_low_rank + # delta_v2
                delta_low_rank * expert_hidden_size +  # delta_u3
                hidden_size * delta_low_rank # delta_v3
            ) +
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )
        inference_moe_decoder_params = (
            # 自注意力参数
            single_attention_params +
            # Router 参数
            single_router_params +
            # shared expert 参数
            shared_expert_params +
            shared_expert_gate_params +
            # MoE 参数
            (
                hidden_size * expert_hidden_size * 2  # Wmean1, Wmean3
            ) +
            num_active_experts * (
                expert_hidden_size * hidden_size + # Wmean2
                delta_low_rank * expert_hidden_size +  # delta_u1
                hidden_size * delta_low_rank + # delta_v1
                delta_low_rank * hidden_size + # delta_u2
                expert_hidden_size * delta_low_rank + # delta_v2
                delta_low_rank * expert_hidden_size +  # delta_u3
                hidden_size * delta_low_rank # delta_v3
            ) +
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )
    elif use_pp == True:
        # 每个 Transformer 层参数
        moe_decoder_params = (
            # 自注意力参数
            single_attention_params +
            # Router 参数
            single_router_params +
            # shared expert 参数
            shared_expert_params +
            shared_expert_gate_params +
            # MoE 参数
            (
                hidden_size * expert_hidden_size * 2 +  # Wmean1, Wmean3
                expert_hidden_size * hidden_size  # Wmean2
            ) * (1 - pp_ratio) +
            num_experts * (
                delta_low_rank * expert_hidden_size +  # delta_u1
                hidden_size * delta_low_rank + # delta_v1
                delta_low_rank * hidden_size + # delta_u2
                expert_hidden_size * delta_low_rank + # delta_v2
                delta_low_rank * expert_hidden_size +  # delta_u3
                hidden_size * delta_low_rank # delta_v3
            ) +
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )
        inference_moe_decoder_params = (
            # 自注意力参数
            single_attention_params +
            # Router 参数
            single_router_params +
            # shared expert 参数
            shared_expert_params +
            shared_expert_gate_params +
            # MoE 参数
            (
                hidden_size * expert_hidden_size * 2  # Wmean1, Wmean3
            ) * (1 - pp_ratio) +
            num_active_experts * (
                expert_hidden_size * hidden_size + # Wmean2
                delta_low_rank * expert_hidden_size +  # delta_u1
                hidden_size * delta_low_rank + # delta_v1
                delta_low_rank * hidden_size + # delta_u2
                expert_hidden_size * delta_low_rank + # delta_v2
                delta_low_rank * expert_hidden_size +  # delta_u3
                hidden_size * delta_low_rank # delta_v3
            ) +
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )
    
    # 所有 Transformer 层的参数
    transformer_params = moe_decoder_params * (num_layers - len(ignore_layers)) + uncompressed_moe_decoder_params * len(ignore_layers)
    inference_transformer_params = inference_moe_decoder_params * (num_layers - len(ignore_layers)) + uncompressed_inference_moe_decoder_params * len(ignore_layers)
    # 输出层参数
    output_params = hidden_size * vocab_size + hidden_size  # lm_head + final_norm
    
    # 总参数量
    total_params = embedding_params + transformer_params + output_params
    inference_total_params = embedding_params + inference_transformer_params + output_params
    # 转换为 Billion
    total_params_b = total_params / (1000 ** 3)
    inference_total_params_b = inference_total_params / (1000 ** 3)
    # 详细分布
    details = {
        "Embedding 层 (B)": embedding_params / (1000 ** 3),
        "所有 Transformer 层 (B)": transformer_params / (1000 ** 3),
        "输出层 (B)": output_params / (1000 ** 3),
        "总参数量 (B)": total_params_b,
        "推理时总参数量 (B)": inference_total_params_b,
    }
    
    return details, total_params_b, inference_total_params_b

def find_delta_deepseek_ratio(target_compression_ratio=0.2, tolerance=1e-4, use_pp=False, share_V=False, pp_ratio=0.2, ignore_layers = [],
                     config=None):
    """
    通过二分查找找到合适的delta_ratio值
    Args:
        target_compression_ratio: 目标压缩比例，如0.2表示压缩20%
        tolerance: 允许的误差范围
    Returns:
        找到的delta_ratio值
    """
    # 获取原始模型参数量
    _, original_params, _ = calculate_deepseek_params(config=config)
    target_params = original_params * (1 - target_compression_ratio)
    
    # 二分查找的范围
    left, right = 0.0, 1.0
    
    while right - left > tolerance:
        mid = (left + right) / 2
        _, compressed_params, _ = calculate_compressed_deepseek_params(delta_ratio=mid, use_pp=use_pp, share_V=share_V, pp_ratio=pp_ratio, config=config)
        
        if compressed_params > target_params:
            # 如果当前参数量太大，需要更小的delta_ratio
            right = mid
        else:
            # 如果当前参数量太小，需要更大的delta_ratio
            left = mid
    
    final_ratio = (left + right) / 2
    _, final_params, _ = calculate_compressed_deepseek_params(delta_ratio=final_ratio, use_pp=use_pp, share_V=share_V, pp_ratio=pp_ratio, config=config)
    
    print(f"Original params: {original_params:.2f}B")
    print(f"Target params: {target_params:.2f}B")
    print(f"Achieved params: {final_params:.2f}B")
    print(f"Compression ratio: {(original_params - final_params) / original_params:.2%}")
    
    return final_ratio

def find_delta_qwen_moe_ratio(target_compression_ratio=0.2, tolerance=1e-4, use_pp=False, share_V=False, pp_ratio=0.2, ignore_layers = [],
                     config=None):
    """
    通过二分查找找到合适的delta_ratio值
    Args:
        target_compression_ratio: 目标压缩比例，如0.2表示压缩20%
        tolerance: 允许的误差范围
    Returns:
        找到的delta_ratio值
    """
    # 获取原始模型参数量
    _, original_params, _ = calculate_qwen_moe_params(config=config)
    target_params = original_params * (1 - target_compression_ratio)
    
    # 二分查找的范围
    left, right = 0.0, 1.0
    
    while right - left > tolerance:
        mid = (left + right) / 2
        _, compressed_params, _ = calculate_compressed_qwen_moe_params(delta_ratio=mid, use_pp=use_pp, share_V=share_V, pp_ratio=pp_ratio, config=config, ignore_layers=ignore_layers)
        
        if compressed_params > target_params:
            # 如果当前参数量太大，需要更小的delta_ratio
            right = mid
        else:
            # 如果当前参数量太小，需要更大的delta_ratio
            left = mid
    
    final_ratio = (left + right) / 2
    _, final_params, _ = calculate_compressed_qwen_moe_params(delta_ratio=final_ratio, use_pp=use_pp, share_V=share_V, pp_ratio=pp_ratio, config=config, ignore_layers=ignore_layers)
    
    print(f"Original params: {original_params:.2f}B")
    print(f"Target params: {target_params:.2f}B")
    print(f"Achieved params: {final_params:.2f}B")
    print(f"Compression ratio: {(original_params - final_params) / original_params:.2%}")
    
    return final_ratio




def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="mixtral", choices=["mixtral", "phi", "deepseek", "qwen"])
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--target_compression_ratio", type=float, default=0.4)
    parser.add_argument("--pp_ratio", type=float, default=0.1)
    args = parser.parse_args()

    if args.model_type == "deepseek":
        deepseek_config = json.load(open(f"{args.model_path}/config.json", "r"))
        delta_ratio = find_delta_deepseek_ratio(target_compression_ratio=0.8, use_pp=True, share_V=False, pp_ratio=0.6, ignore_layers = [], config=deepseek_config)
    elif args.model_type == "qwen":
        qwen_config = json.load(open(f"{args.model_path}/config.json", "r"))
        delta_ratio = find_delta_qwen_moe_ratio(target_compression_ratio=0.4, use_pp=True, share_V=False, pp_ratio=0.1, config=qwen_config, ignore_layers=[12,13,14,15])
    elif args.model_type == "mixtral":
        mixtral_config = json.load(open(f"{args.model_path}/config.json", "r"))
        delta_ratio = find_delta_ratio(target_compression_ratio=0.8, use_pp=True, share_V=False, pp_ratio=0.6, ignore_layers = [], config=mixtral_config)
        
    elif args.model_type == "phi":
        phi_config = json.load(open(f"{args.model_path}/config.json", "r"))
        delta_ratio = find_delta_ratio(target_compression_ratio=0.8, use_pp=True, share_V=False, pp_ratio=0.6, ignore_layers = [], config=phi_config)


    
    print(f"Found pruning_ratio: {args.pp_ratio:.4f}")
    print(f"Found delta_ratio: {delta_ratio:.4f}")


if __name__ == "__main__":
    main()