from transformers.models.mixtral.modeling_mixtral import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from pathlib import Path
import json
from tqdm import tqdm
import os
from accelerate import init_empty_weights
from functools import partial
import random
from torch.utils.data import DataLoader
from datasets import load_dataset


def cal_scale_inv(svd_scale):
    try:
        scale_inv = torch.linalg.inv(svd_scale.to(torch.float32))
    except Exception as e:
        print("Warning: svd_scale is not full rank!")
        svd_scale += 1e-6 * torch.eye(svd_scale.shape[0]).to(svd_scale.device)
        scale_inv = torch.linalg.inv(svd_scale)
    return scale_inv.float()


class Merge_MixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config, share_ratio, delta_ratio, expert_freq, delta_share_V=False, delta_share_U=False, merge_method="freq",  shared_infer=False):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.intermediate_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.dtype = torch.bfloat16

        # added
        self.share_ratio = share_ratio
        self.delta_ratio = delta_ratio
        self.expert_freq = expert_freq
        self.config = config

        self.delta_share_V = delta_share_V
        self.delta_share_U = delta_share_U
        self.merge_method = merge_method
        self.shared_infer = shared_infer
        self.act_fn = ACT2FN[config.hidden_act]

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False, dtype=torch.bfloat16)
        
        self.Wmean1 = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False, dtype=self.dtype)
        self.Wmean2 = nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False, dtype=self.dtype)
        self.Wmean3 = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False, dtype=self.dtype)

        if self.delta_share_V == False and self.delta_share_U == False:
            self.experts = nn.ModuleList([meanW_deltaUV(config, self.Wmean1, self.Wmean2, self.Wmean3, 
                                                    self.delta_ratio, delta_share_V=False, delta_share_U=False)  for _ in range(self.num_experts)])
            
        elif self.delta_share_V == True and self.delta_share_U == False:
            if self.delta_ratio != 0:
                delta_low_rank = int(self.intermediate_dim * self.hidden_dim * self.delta_ratio / (self.intermediate_dim + self.hidden_dim))
                self.experts_delta_v1_shared = nn.Linear(self.hidden_dim, delta_low_rank, bias=False, dtype=torch.bfloat16)
                self.experts_delta_v2_shared = nn.Linear(self.intermediate_dim, delta_low_rank, bias=False, dtype=torch.bfloat16)
                self.experts_delta_v3_shared = nn.Linear(self.hidden_dim, delta_low_rank, bias=False, dtype=torch.bfloat16)
            else:
                self.experts_delta_v1_shared = None
                self.experts_delta_v2_shared = None
                self.experts_delta_v3_shared = None
            self.experts = nn.ModuleList([meanW_deltaUV(config, self.Wmean1, self.Wmean2, self.Wmean3, 
                                                    self.delta_ratio, delta_share_V=True, delta_share_U=False, 
                                                    experts_delta_v1_shared=self.experts_delta_v1_shared, 
                                                    experts_delta_v2_shared=self.experts_delta_v2_shared, experts_delta_v3_shared=self.experts_delta_v3_shared)  for _ in range(self.num_experts)])
            
        elif self.delta_share_V == True and self.delta_share_U == True:
            if self.delta_ratio != 0:
                delta_low_rank = int(self.intermediate_dim * self.hidden_dim * self.delta_ratio / (self.intermediate_dim + self.hidden_dim))
                self.experts_delta_u1_shared = nn.Linear(delta_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
                self.experts_delta_v1_shared = nn.Linear(self.hidden_dim, delta_low_rank, bias=False, dtype=torch.bfloat16)
                self.experts_delta_u2_shared = nn.Linear(delta_low_rank, self.hidden_dim, bias=False, dtype=torch.bfloat16)
                self.experts_delta_v2_shared = nn.Linear(self.intermediate_dim, delta_low_rank, bias=False, dtype=torch.bfloat16)
                self.experts_delta_u3_shared = nn.Linear(delta_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
                self.experts_delta_v3_shared = nn.Linear(self.hidden_dim, delta_low_rank, bias=False, dtype=torch.bfloat16)
            else:
                self.experts_delta_u1_shared = None
                self.experts_delta_v1_shared = None
                self.experts_delta_u2_shared = None
                self.experts_delta_v2_shared = None
                self.experts_delta_u3_shared = None
                self.experts_delta_v3_shared = None
            self.experts = nn.ModuleList([meanW_deltaUV(config, self.Wmean1, self.Wmean2, self.Wmean3, 
                                                    self.delta_ratio, delta_share_V=True, delta_share_U=True, experts_delta_u1_shared=self.experts_delta_u1_shared, experts_delta_v1_shared=self.experts_delta_v1_shared, 
                                                    experts_delta_u2_shared=self.experts_delta_u2_shared, experts_delta_v2_shared=self.experts_delta_v2_shared, 
                                                    experts_delta_u3_shared=self.experts_delta_u3_shared, experts_delta_v3_shared=self.experts_delta_v3_shared)  for _ in range(self.num_experts)])

        self.jitter_noise = config.router_jitter_noise


    def update_Wmean(self):
        for i in range(self.num_experts):
            self.experts[i].Wmean1 = self.Wmean1
            self.experts[i].Wmean2 = self.Wmean2
            self.experts[i].Wmean3 = self.Wmean3


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        if self.shared_infer == True:
            if cfg['test_stage'] == True:
                up_hidden_states = self.Wmean3(hidden_states)
                gate_hidden_states = self.Wmean1(hidden_states)
            elif cfg['no_probe_process'] == True:
                keep_dim = int(self.hidden_dim * (1 - cfg['prune_ratio']))
                probe_out_dim_indices = torch.arange(keep_dim)
                up_hidden_states = self.Wmean3(hidden_states, out_dim_indices=probe_out_dim_indices)
                gate_hidden_states = self.Wmean1(hidden_states, out_dim_indices=probe_out_dim_indices)
                down_hidden_states = self.Wmean2(self.act_fn(gate_hidden_states) * up_hidden_states, in_dim_indices=probe_out_dim_indices)
            
            
            
            
            if self.delta_share_V == True and self.delta_ratio != 0:
                up_deltav_hidden_states = self.experts_delta_v3_shared(hidden_states)
                gate_deltav_hidden_states = self.experts_delta_v1_shared(hidden_states)
                if cfg['no_probe_process'] == True:
                    down_deltav_hidden_states = F.linear( (self.act_fn(gate_hidden_states) * up_hidden_states), self.experts_delta_v2_shared.weight[:, probe_out_dim_indices], bias=None)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.numel() > 0:
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                
                if self.shared_infer == True:
                    current_up_hidden_states = up_hidden_states[top_x]
                    current_gate_hidden_states = gate_hidden_states[top_x]

                    if cfg['no_probe_process'] == True:
                        current_down_hidden_states = down_hidden_states[top_x]
                    else:
                        current_down_hidden_states = None


                    if self.delta_share_V == True and self.delta_ratio != 0:
                        current_up_deltav_hidden_states = up_deltav_hidden_states[top_x]
                        current_gate_deltav_hidden_states = gate_deltav_hidden_states[top_x]
                        if cfg['no_probe_process'] == True:
                            current_down_deltav_hidden_states = down_deltav_hidden_states[top_x]
                        else:
                            current_down_deltav_hidden_states = None
                    
                        current_hidden_states = expert_layer(current_state, current_up_hidden_states, current_gate_hidden_states, current_down_hidden_states,
                                                            current_up_deltav_hidden_states, current_gate_deltav_hidden_states, current_down_deltav_hidden_states, 
                                                            shared_infer=self.shared_infer) * routing_weights[top_x, idx, None]
                    else:
                        current_hidden_states = expert_layer(current_state, current_up_hidden_states, current_gate_hidden_states, current_down_hidden_states, 
                                                             shared_infer=self.shared_infer) * routing_weights[top_x, idx, None]
                else:
                    current_hidden_states = expert_layer(current_state, shared_infer=self.shared_infer) * routing_weights[top_x, idx, None]

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    @staticmethod
    @torch.no_grad()
    def svd_delta(W, ratio=1, svd_scale=None, rank=None, absorb_u=False, absorb_v=False, scale_type='svdllm'):
        if rank is None:
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        else:
            num_s_after_trunc = rank
        if svd_scale is None:
            U, S, VT = torch.linalg.svd(W.float(), full_matrices=False)
            del W
            truc_s = S[:num_s_after_trunc]
            del S
            truc_u = U[:, :num_s_after_trunc]
            del U
            truc_v = VT[:num_s_after_trunc, :]
            del VT
            truc_sigma = torch.diag(truc_s)
            del truc_s
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma)
            svd_v = torch.matmul(sqrtSigma, truc_v)
        else:
            if scale_type == 'svdllm':
                W_scale = torch.matmul(W, svd_scale.bfloat16().to(W.device))
                U, S, VT = torch.linalg.svd(W_scale.float(), full_matrices=False)
                del W_scale
                truc_s = S[:num_s_after_trunc]
                del S
                truc_u = U[:, :num_s_after_trunc]
                del U
                truc_v = torch.matmul(VT[:num_s_after_trunc, :], cal_scale_inv(svd_scale).to(W.device))
                del VT
                truc_sigma = torch.diag(truc_s)
                del truc_s
                if absorb_u:
                    svd_u = torch.matmul(truc_u, truc_sigma)
                    svd_v = truc_v
                elif absorb_v:
                    svd_u = truc_u
                    svd_v = torch.matmul(truc_sigma, truc_v)
                else:
                    sqrtSigma = torch.sqrt(truc_sigma)
                    svd_u = torch.matmul(truc_u, sqrtSigma)
                    svd_v = torch.matmul(sqrtSigma, truc_v)
            elif scale_type == 'asvd':
                alpha = 0.5
                svd_scale *= alpha
                svd_scale += 1e-6
                W_scale = W * svd_scale.to(W.device).view(1, -1)
                U, S, VT = torch.linalg.svd(W_scale.float(), full_matrices=False)
                del W_scale
                truc_s = S[:num_s_after_trunc]
                del S
                truc_u = U[:, :num_s_after_trunc]
                del U
                VT= VT / svd_scale.to(W.device).view(-1, 1)
                truc_v = VT[:num_s_after_trunc, :]
                del VT
                truc_sigma = torch.diag(truc_s)
                del truc_s
                sqrtSigma = torch.sqrt(truc_sigma)
                svd_u = torch.matmul(truc_u, sqrtSigma)
                svd_v = torch.matmul(sqrtSigma, truc_v)
        return svd_u.to(torch.bfloat16), svd_v.to(torch.bfloat16)


    @torch.no_grad()
    def merge_experts(self, module, svd_scale = None, hessian = None, scale_type='svdllm', preprocess_method = None):
        self.gate.weight.data = module.gate.weight.data




        if preprocess_method == "permutation-weight":
            import numpy as np
            domain_idx = np.argmax(self.expert_freq)

            from scipy.optimize import linear_sum_assignment
            @torch.no_grad()
            def compute_switch_permutation_by_weight_matching(reference_expert,target_expert) -> torch.Tensor:
                reference_w1 = reference_expert.w1.weight
                reference_w2 = reference_expert.w2.weight
                reference_w3 = reference_expert.w3.weight

                target_w1 = target_expert.w1.weight
                target_w2 = target_expert.w2.weight
                target_w3 = target_expert.w3.weight

                lsa_cost_matrix = torch.mm(reference_w1.data, target_w1.data.t())
                lsa_cost_matrix += torch.mm(reference_w2.data, target_w2.data.t())
                lsa_cost_matrix += torch.mm(reference_w3.data, target_w3.data.t())

                _, perm = linear_sum_assignment(lsa_cost_matrix.cpu().numpy(), maximize=True)
                return torch.from_numpy(perm).to(lsa_cost_matrix.device)

            @torch.no_grad()
            def permute_switch_mlp_dense_expert_(expert, perm):
                expert.w1.weight.data = expert.w1.weight.data[perm, :]
                expert.w2.weight.data = expert.w2.weight.data[:, perm]
                expert.w3.weight.data = expert.w3.weight.data[perm, :]
                return expert

            for i in range(len(module.experts)):
                if i == domain_idx:
                    continue
                perm = compute_switch_permutation_by_weight_matching(
                    module.experts[domain_idx],
                    module.experts[i],
                )
                module.experts[i] = permute_switch_mlp_dense_expert_(
                    module.experts[i], perm
                )

        if self.merge_method == "freq":
            self.Wmean1.weight.data = sum([module.experts[i].w1.weight * self.expert_freq[i] for i in range(self.num_experts)]) / sum(self.expert_freq)
            self.Wmean2.weight.data = sum([module.experts[i].w2.weight * self.expert_freq[i] for i in range(self.num_experts)]) / sum(self.expert_freq)
            self.Wmean3.weight.data = sum([module.experts[i].w3.weight * self.expert_freq[i] for i in range(self.num_experts)]) / sum(self.expert_freq)
        elif self.merge_method == "mean":
            self.Wmean1.weight.data = sum([module.experts[i].w1.weight for i in range(self.num_experts)]) / self.num_experts
            self.Wmean2.weight.data = sum([module.experts[i].w2.weight for i in range(self.num_experts)]) / self.num_experts
            self.Wmean3.weight.data = sum([module.experts[i].w3.weight for i in range(self.num_experts)]) / self.num_experts

        elif self.merge_method == "fisher":
            scaling_factor = 1.0
            scaling_factors_ft_list = self.expert_freq

            fisher_scale_w1 = [0]*8
            fisher_scale_w2 = [0]*8
            fisher_scale_w3 = [0]*8

            for j in range(8):
                base_name = f"block_sparse_moe.experts.{j}."
                fisher_scale_w1[j] = hessian[base_name + "w1"].to(module.experts[j].w1.weight.device) * scaling_factors_ft_list[j]
                fisher_scale_w2[j] = hessian[base_name + "w2"].to(module.experts[j].w2.weight.device) * scaling_factors_ft_list[j]
                fisher_scale_w3[j] = hessian[base_name + "w3"].to(module.experts[j].w3.weight.device) * scaling_factors_ft_list[j]

            fisher_scale_w1 = [fisher_scale_w1[j] / sum(fisher_scale_w1) for j in range(8)]
            fisher_scale_w2 = [fisher_scale_w2[j] / sum(fisher_scale_w2) for j in range(8)]
            fisher_scale_w3 = [fisher_scale_w3[j] / sum(fisher_scale_w3) for j in range(8)]
                
            self.Wmean1.weight.data = sum([module.experts[j].w1.weight * fisher_scale_w1[j] for j in range(8)])
            self.Wmean2.weight.data = sum([module.experts[j].w2.weight * fisher_scale_w2[j] for j in range(8)])
            self.Wmean3.weight.data = sum([module.experts[j].w3.weight * fisher_scale_w3[j] for j in range(8)])       

        elif self.merge_method == "random":
            torch.manual_seed(111)
            scale_alpha_1 = torch.rand(8, device=module.experts[0].w1.weight.device)
            torch.manual_seed(222)
            scale_alpha_2 = torch.rand(8, device=module.experts[0].w1.weight.device)
            torch.manual_seed(333)
            scale_alpha_3 = torch.rand(8, device=module.experts[0].w1.weight.device)
            scale_alpha_1 = scale_alpha_1 / scale_alpha_1.sum()  # Normalize to sum to 1
            scale_alpha_2 = scale_alpha_2 / scale_alpha_2.sum()  # Normalize to sum to 1
            scale_alpha_3 = scale_alpha_3 / scale_alpha_3.sum()  # Normalize to sum to 1

            self.Wmean1.weight.data = sum([module.experts[i].w1.weight * scale_alpha_1[i] for i in range(8)])
            self.Wmean2.weight.data = sum([module.experts[i].w2.weight * scale_alpha_2[i] for i in range(8)])
            self.Wmean3.weight.data = sum([module.experts[i].w3.weight * scale_alpha_3[i] for i in range(8)])

        elif self.merge_method == "regmean":
            grams = hessian
            def reduce_non_diag(cov_mat, a):
                diag_weight = torch.diag(torch.ones(cov_mat.size(0)) - a).to(cov_mat.device)
                non_diag_weight = torch.zeros_like(diag_weight).fill_(a)
                weight = diag_weight + non_diag_weight
                ret = cov_mat * weight
                return ret
            
            for _, gram in grams.items():
                gram = reduce_non_diag(gram, a=0.5)

            sum_w1 = 0
            sum_w2 = 0
            sum_w3 = 0
            sum_gram_w1 = 0
            sum_gram_w2 = 0
            sum_gram_w3 = 0
            for j in range(8):
                base_name = f"block_sparse_moe.experts.{j}."
                sum_w1 += torch.matmul(grams[base_name + "w1"], module.experts[j].w1.weight.transpose(0,1))
                sum_w2 += torch.matmul(grams[base_name + "w2"], module.experts[j].w2.weight.transpose(0,1))
                sum_w3 += torch.matmul(grams[base_name + "w3"], module.experts[j].w3.weight.transpose(0,1))

                sum_gram_w1 += grams[base_name + "w1"]
                sum_gram_w2 += grams[base_name + "w2"]
                sum_gram_w3 += grams[base_name + "w3"]

            self.Wmean1.weight.data = torch.matmul(torch.inverse(sum_gram_w1.to(torch.float32)).to(torch.bfloat16), sum_w1).transpose(0,1)
            self.Wmean2.weight.data = torch.matmul(torch.inverse(sum_gram_w2.to(torch.float32)).to(torch.bfloat16), sum_w2).transpose(0,1)
            self.Wmean3.weight.data = torch.matmul(torch.inverse(sum_gram_w3.to(torch.float32)).to(torch.bfloat16), sum_w3).transpose(0,1)
        
        elif self.merge_method == "ties":
            lamda = 1
            import numpy as np
            domain_idx = np.argmax(self.expert_freq)

            def to_vector(tensor):
                return tensor.flatten()

            def topk_values_mask(M, K=0.7, return_mask=False):
                if K > 1:
                    K /= 100

                original_shape = M.shape
                if M.dim() == 1:
                    M = M.unsqueeze(0)

                n, d = M.shape
                k = int(d * K)
                k = d - k  # Keep top k elements instead of bottom k elements

                # Find the k-th smallest element by magnitude for each row
                kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
                # Create a mask tensor with True for the top k elements in each row
                mask = M.abs() >= kth_values
                final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

                if return_mask:
                    return M * final_mask, final_mask.float().mean(dim=1), final_mask
                return M * final_mask
            def resolve_zero_signs(sign_to_mult, method="majority"):
                majority_sign = torch.sign(sign_to_mult.sum())

                if method == "majority":
                    sign_to_mult[sign_to_mult == 0] = majority_sign
                elif method == "minority":
                    sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
                return sign_to_mult

            def resolve_sign(Tensor):
                sign_to_mult = torch.sign(Tensor.sum(dim=0))
                sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
                return sign_to_mult

            def disjoint_merge(Tensor, merge_func, sign_to_mult):

                merge_func = merge_func.split("-")[-1]

                # If sign is provided then we select the corresponding entries and aggregate.
                if sign_to_mult is not None:
                    rows_to_keep = torch.where(
                        sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
                    )
                    selected_entries = Tensor * rows_to_keep
                # Else we select all non-zero entries and aggregate.
                else:
                    rows_to_keep = Tensor != 0
                    selected_entries = Tensor * rows_to_keep

                if merge_func == "mean":
                    non_zero_counts = (selected_entries != 0).sum(dim=0).float()
                    disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
                        non_zero_counts, min=1
                    )
                elif merge_func == "sum":
                    disjoint_aggs = torch.sum(selected_entries, dim=0)
                elif merge_func == "max":
                    disjoint_aggs = selected_entries.abs().max(dim=0)[0]
                    disjoint_aggs *= sign_to_mult
                else:
                    raise ValueError(f"Merge method {merge_func} is not defined.")

                return disjoint_aggs

            delta_w1 = torch.zeros(7, module.experts[0].w1.weight.shape[0] * module.experts[0].w1.weight.shape[1], device=module.experts[0].w1.weight.device)
            delta_w2 = torch.zeros(7, module.experts[0].w2.weight.shape[0] * module.experts[0].w2.weight.shape[1], device=module.experts[0].w2.weight.device)
            delta_w3 = torch.zeros(7, module.experts[0].w3.weight.shape[0] * module.experts[0].w3.weight.shape[1], device=module.experts[0].w3.weight.device)
            idx = 0
            for j in range(8):
                if j == domain_idx:
                    continue
                delta_w1[idx] = to_vector(module.experts[j].w1.weight - module.experts[domain_idx].w1.weight)
                delta_w2[idx] = to_vector(module.experts[j].w2.weight - module.experts[domain_idx].w2.weight)
                delta_w3[idx] = to_vector(module.experts[j].w3.weight - module.experts[domain_idx].w3.weight)
                idx += 1

            update_w1 = topk_values_mask(delta_w1, K=20)
            update_w2 = topk_values_mask(delta_w2, K=20)
            update_w3 = topk_values_mask(delta_w3, K=20)

            final_sign_w1 = resolve_sign(update_w1)
            final_sign_w2 = resolve_sign(update_w2)
            final_sign_w3 = resolve_sign(update_w3)

            merged_w1 = disjoint_merge(update_w1, "max", final_sign_w1)
            merged_w2 = disjoint_merge(update_w2, "max", final_sign_w2)
            merged_w3 = disjoint_merge(update_w3, "max", final_sign_w3)

            self.Wmean1.weight.data = (module.experts[domain_idx].w1.weight + lamda * merged_w1.reshape(module.experts[domain_idx].w1.weight.shape)).to(torch.bfloat16)
            self.Wmean2.weight.data = (module.experts[domain_idx].w2.weight + lamda * merged_w2.reshape(module.experts[domain_idx].w2.weight.shape)).to(torch.bfloat16)
            self.Wmean3.weight.data = (module.experts[domain_idx].w3.weight + lamda * merged_w3.reshape(module.experts[domain_idx].w3.weight.shape)).to(torch.bfloat16)            
        
        else:
            raise ValueError(f"wrong merge method {self.merge_method}!")


        if self.delta_ratio != 0:
            scale_w1_mean = None
            scale_w2_mean = None
            scale_w3_mean = None
            total_freq = 0
            if svd_scale is not None:
                for j in range(self.num_experts):
                    base_name = f"block_sparse_moe.experts.{j}."
                    freq = self.expert_freq[j]  
                    total_freq += freq
                    if scale_w1_mean is None:
                        scale_w1_mean = svd_scale[base_name + "w1"] * freq
                    else:
                        scale_w1_mean += svd_scale[base_name + "w1"] * freq
                    if scale_w2_mean is None:
                        scale_w2_mean = svd_scale[base_name + "w2"] * freq
                    else:
                        scale_w2_mean += svd_scale[base_name + "w2"] * freq
                    if scale_w3_mean is None:
                        scale_w3_mean = svd_scale[base_name + "w3"] * freq
                    else:
                        scale_w3_mean += svd_scale[base_name + "w3"] * freq
                scale_w1_mean /= total_freq
                scale_w2_mean /= total_freq
                scale_w3_mean /= total_freq

            if self.delta_share_V == True and self.delta_share_U == False:
                delta_w1 = []
                delta_w2 = []
                delta_w3 = []

                for j in tqdm(range(self.num_experts), desc="Merging experts", leave=False):
                    
                    delta_w1.append(module.experts[j].w1.weight - self.Wmean1.weight)
                    delta_w2.append(module.experts[j].w2.weight - self.Wmean2.weight)
                    delta_w3.append(module.experts[j].w3.weight - self.Wmean3.weight)

                delta_w1 = torch.stack(delta_w1, dim=0).reshape(-1, delta_w1[0].shape[1])
                delta_w2 = torch.stack(delta_w2, dim=0).reshape(-1, delta_w2[0].shape[1])
                delta_w3 = torch.stack(delta_w3, dim=0).reshape(-1, delta_w3[0].shape[1])

                if svd_scale is None:
                    delta_u1, shared_v1 = self.svd_delta(delta_w1, rank=self.experts[0].delta_low_rank)
                    delta_u2, shared_v2 = self.svd_delta(delta_w2, rank=self.experts[0].delta_low_rank)
                    delta_u3, shared_v3 = self.svd_delta(delta_w3, rank=self.experts[0].delta_low_rank)
                else:
                    delta_u1, shared_v1 = self.svd_delta(delta_w1, rank=self.experts[0].delta_low_rank, svd_scale=scale_w1_mean, scale_type=scale_type)
                    delta_u2, shared_v2 = self.svd_delta(delta_w2, rank=self.experts[0].delta_low_rank, svd_scale=scale_w2_mean, scale_type=scale_type)
                    delta_u3, shared_v3 = self.svd_delta(delta_w3, rank=self.experts[0].delta_low_rank, svd_scale=scale_w3_mean, scale_type=scale_type)

                shared_v1 = nn.Parameter(shared_v1)
                shared_v2 = nn.Parameter(shared_v2)
                shared_v3 = nn.Parameter(shared_v3)

                del delta_w1, delta_w2, delta_w3

                self.experts_delta_v1_shared.weight = shared_v1
                self.experts_delta_v2_shared.weight = shared_v2
                self.experts_delta_v3_shared.weight = shared_v3

                for j in tqdm(range(self.num_experts), desc="Merging experts", leave=False):

                    self.experts[j].delta_u1.weight.data = delta_u1[j * self.experts[j].delta_u1.weight.shape[0]:(j + 1) * self.experts[j].delta_u1.weight.shape[0], :]
                    self.experts[j].delta_u2.weight.data = delta_u2[j * self.experts[j].delta_u2.weight.shape[0]:(j + 1) * self.experts[j].delta_u2.weight.shape[0], :]
                    self.experts[j].delta_u3.weight.data = delta_u3[j * self.experts[j].delta_u3.weight.shape[0]:(j + 1) * self.experts[j].delta_u3.weight.shape[0], :]

            if self.delta_share_V == True and self.delta_share_U == True:
                delta_w1 = []
                delta_w2 = []
                delta_w3 = []

                for j in tqdm(range(self.num_experts), desc="Merging experts", leave=False):               
                    delta_w1.append(module.experts[j].w1.weight - self.Wmean1.weight)
                    delta_w2.append(module.experts[j].w2.weight - self.Wmean2.weight)
                    delta_w3.append(module.experts[j].w3.weight - self.Wmean3.weight)

                delta_w1 = torch.stack(delta_w1, dim=0).reshape(-1, delta_w1[0].shape[1])
                delta_w2 = torch.stack(delta_w2, dim=0).reshape(-1, delta_w2[0].shape[1])
                delta_w3 = torch.stack(delta_w3, dim=0).reshape(-1, delta_w3[0].shape[1])

                if svd_scale is None:
                    delta_u1, shared_v1 = self.svd_delta(delta_w1, rank=self.experts[0].delta_low_rank)
                    delta_u2, shared_v2 = self.svd_delta(delta_w2, rank=self.experts[0].delta_low_rank)
                    delta_u3, shared_v3 = self.svd_delta(delta_w3, rank=self.experts[0].delta_low_rank)
                else:
                    delta_u1, shared_v1 = self.svd_delta(delta_w1, rank=self.experts[0].delta_low_rank, svd_scale=scale_w1_mean, scale_type=scale_type)
                    delta_u2, shared_v2 = self.svd_delta(delta_w2, rank=self.experts[0].delta_low_rank, svd_scale=scale_w2_mean, scale_type=scale_type)
                    delta_u3, shared_v3 = self.svd_delta(delta_w3, rank=self.experts[0].delta_low_rank, svd_scale=scale_w3_mean, scale_type=scale_type)

                shared_v1 = nn.Parameter(shared_v1)
                shared_v2 = nn.Parameter(shared_v2)
                shared_v3 = nn.Parameter(shared_v3)

                shared_u1 = None
                shared_u2 = None
                shared_u3 = None
                total_freq = 0
                for j in range(self.num_experts):
                    freq = self.expert_freq[j]
                    total_freq += freq
                    if shared_u1 is None:
                        shared_u1 = delta_u1[j * self.experts[j].delta_u1.weight.shape[0]:(j + 1) * self.experts[j].delta_u1.weight.shape[0], :] * freq 
                    else:
                        shared_u1 += delta_u1[j * self.experts[j].delta_u1.weight.shape[0]:(j + 1) * self.experts[j].delta_u1.weight.shape[0], :] * freq
                    if shared_u2 is None:
                        shared_u2 = delta_u2[j * self.experts[j].delta_u2.weight.shape[0]:(j + 1) * self.experts[j].delta_u2.weight.shape[0], :] * freq
                    else:
                        shared_u2 += delta_u2[j * self.experts[j].delta_u2.weight.shape[0]:(j + 1) * self.experts[j].delta_u2.weight.shape[0], :] * freq
                    if shared_u3 is None:
                        shared_u3 = delta_u3[j * self.experts[j].delta_u3.weight.shape[0]:(j + 1) * self.experts[j].delta_u3.weight.shape[0], :] * freq
                    else:
                        shared_u3 += delta_u3[j * self.experts[j].delta_u3.weight.shape[0]:(j + 1) * self.experts[j].delta_u3.weight.shape[0], :] * freq
                shared_u1 /= total_freq
                shared_u2 /= total_freq
                shared_u3 /= total_freq

                shared_u1 = nn.Parameter(shared_u1)
                shared_u2 = nn.Parameter(shared_u2)
                shared_u3 = nn.Parameter(shared_u3)

                self.experts_delta_u1_shared.weight = shared_u1
                self.experts_delta_v1_shared.weight = shared_v1
                self.experts_delta_u2_shared.weight = shared_u2
                self.experts_delta_v2_shared.weight = shared_v2
                self.experts_delta_u3_shared.weight = shared_u3
                self.experts_delta_v3_shared.weight = shared_v3


            if self.delta_share_V == False and self.delta_share_U == False:
                for j in tqdm(range(self.num_experts), desc="Merging experts", leave=False):
                    delta_w1 = (module.experts[j].w1.weight - self.Wmean1.weight)
                    delta_w2 = (module.experts[j].w2.weight - self.Wmean2.weight)
                    delta_w3 = (module.experts[j].w3.weight - self.Wmean3.weight)

                    if svd_scale is not None:
                        base_name = f"block_sparse_moe.experts.{j}."
                        self.experts[j].delta_u1.weight.data, self.experts[j].delta_v1.weight.data = self.svd_delta(delta_w1, ratio=self.delta_ratio, svd_scale=svd_scale[base_name + "w1"], scale_type=scale_type)
                        self.experts[j].delta_u2.weight.data, self.experts[j].delta_v2.weight.data = self.svd_delta(delta_w2, ratio=self.delta_ratio, svd_scale=svd_scale[base_name + "w2"], scale_type=scale_type)
                        self.experts[j].delta_u3.weight.data, self.experts[j].delta_v3.weight.data = self.svd_delta(delta_w3, ratio=self.delta_ratio, svd_scale=svd_scale[base_name + "w3"], scale_type=scale_type)
                    else:
                        self.experts[j].delta_u1.weight.data, self.experts[j].delta_v1.weight.data = self.svd_delta(delta_w1, ratio=self.delta_ratio)
                        self.experts[j].delta_u2.weight.data, self.experts[j].delta_v2.weight.data = self.svd_delta(delta_w2, ratio=self.delta_ratio)
                        self.experts[j].delta_u3.weight.data, self.experts[j].delta_v3.weight.data = self.svd_delta(delta_w3, ratio=self.delta_ratio)
            del svd_scale


 
from config import cfg
from .pruning_module import HiddenRepresentationPruning
from hf.utils import generate_probe, check_nan_inf, get_next_layer  

class meanW_deltaUV(nn.Module):
    def __init__(self, config: MixtralConfig, Wmean1, Wmean2, Wmean3, delta_ratio=1, delta_share_V=False, delta_share_U=False, 
                 experts_delta_u1_shared=None, experts_delta_v1_shared=None, 
                 experts_delta_u2_shared=None, experts_delta_v2_shared=None, 
                 experts_delta_u3_shared=None, experts_delta_v3_shared=None, layer_order=[]):
        super().__init__()
        self.intermediate_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.dtype = torch.bfloat16
        self.delta_share_V = delta_share_V
        self.delta_share_U = delta_share_U

        self.Wmean1 = Wmean1
        self.Wmean2 = Wmean2
        self.Wmean3 = Wmean3
            
        self.delta_ratio = delta_ratio

        self.act_fn = ACT2FN[config.hidden_act]

        if delta_share_V == False and delta_share_U == False and self.delta_ratio != 0:
            self.delta_ratio = delta_ratio
            self.delta_low_rank = int(self.intermediate_dim * self.hidden_dim * self.delta_ratio / (self.intermediate_dim + self.hidden_dim))
            self.delta_u1 = nn.Linear(self.delta_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
            self.delta_v1 = nn.Linear(self.hidden_dim, self.delta_low_rank, bias=False, dtype=torch.bfloat16)
            self.delta_u2 = nn.Linear(self.delta_low_rank, self.hidden_dim, bias=False, dtype=torch.bfloat16)
            self.delta_v2 = nn.Linear(self.intermediate_dim, self.delta_low_rank, bias=False, dtype=torch.bfloat16)
            self.delta_u3 = nn.Linear(self.delta_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
            self.delta_v3 = nn.Linear(self.hidden_dim, self.delta_low_rank, bias=False, dtype=torch.bfloat16)

        elif delta_share_V == True and delta_share_U == False and self.delta_ratio != 0:
            self.delta_ratio = delta_ratio
            self.delta_low_rank = int(self.intermediate_dim * self.hidden_dim * self.delta_ratio / (self.intermediate_dim + self.hidden_dim))
            self.delta_u1 = nn.Linear(self.delta_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
            self.delta_v1 = experts_delta_v1_shared
            self.delta_u2 = nn.Linear(self.delta_low_rank, self.hidden_dim, bias=False, dtype=torch.bfloat16)
            self.delta_v2 = experts_delta_v2_shared
            self.delta_u3 = nn.Linear(self.delta_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
            self.delta_v3 = experts_delta_v3_shared
        
        elif delta_share_V == True and delta_share_U == True and self.delta_ratio != 0:
            self.delta_u1 = experts_delta_u1_shared
            self.delta_v1 = experts_delta_v1_shared
            self.delta_u2 = experts_delta_u2_shared
            self.delta_v2 = experts_delta_v2_shared
            self.delta_u3 = experts_delta_u3_shared
            self.delta_v3 = experts_delta_v3_shared

        # --------------------------------------------------
        self.layer_order = layer_order
        self.pruning_module = HiddenRepresentationPruning(cfg, f'Mixtral_mlp_{layer_order}', config)
        self.probe_out_dim_indices = None
        # --------------------------------------------------
        

    def probe_process(self, x, **kwargs):
        # 1. generate probeW
        # 2. run matrix multiplication
        # 3. calculate score
        # 4. extract metric

        # generate probe
        # rank / mean / absnml
        # cur_batch_seq_len = x.size(1)
        cur_batch_seq_len = x.size(0)
        if cfg['gate_probe_ratio'] == cfg['up_probe_ratio']:
            # if 'respick' in cfg['prune_method']:
            #     residual_for_probe = kwargs['respick']
            # else:
            #     residual_for_probe = None
            residual_for_probe = None
            probe, seq_selected_indices = generate_probe(x, cfg['gate_probe_ratio'], residual_for_probe)
        else:
            raise ValueError('gate_probe_num should be equal to up_probe_num for now')
        
        probe_out = self.act_fn(self.Wmean1(probe, cal_mlp_probe_out_dim_metric=True)) * self.Wmean3(probe, cal_mlp_probe_out_dim_metric=True) + self.act_fn(self.delta_u1(self.delta_v1(probe)) * self.delta_u3(self.delta_v3(probe)))

        # calculate score
        if 'calib' in cfg['prune_method'] or 'runningmean' in cfg['prune_method'] or 'ema' in cfg['prune_method']:
            probe_out_dim_metric = self.pruning_module.cal_mlp_prune_metric(probe_out, self.Wmean2.weight.data + torch.matmul(self.delta_u2.weight.data, self.delta_v2.weight.data), cfg['prune_metric'], seq_selected_indices, global_metric_score_distribution=self.Wmean2.get_global_metric_score_distribution(cur_batch_seq_len))
        else:
            probe_out_dim_metric = self.pruning_module.cal_mlp_prune_metric(probe_out, self.Wmean2.weight.data + torch.matmul(self.delta_u2.weight.data, self.delta_v2.weight.data), cfg['prune_metric'], seq_selected_indices)

        if 'flapratio' in cfg['prune_method']:
            probe_out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(probe_out_dim_metric, cfg['tc_multiple'], pruning_ratio=self.Wmean2.pruning_ratio)
        else:
            probe_out_dim_indices, prune_out_dim_indices = self.pruning_module.sort_mlp_metric(probe_out_dim_metric, cfg['tc_multiple'])

        # extract matrix
        self.Wmean1.prepare_async_weight(out_dim_indices=probe_out_dim_indices)
        self.Wmean3.prepare_async_weight(out_dim_indices=probe_out_dim_indices)
        self.Wmean2.prepare_async_weight(in_dim_indices=probe_out_dim_indices)
        return probe_out_dim_indices, probe_out


    def forward(self, x, current_up_hidden_states = None, current_gate_hidden_states = None, current_down_hidden_states = None,
                current_up_deltav_hidden_states = None, current_gate_deltav_hidden_states = None, current_down_deltav_hidden_states = None,
                shared_infer=False, **kwargs):
        if cfg['test_stage'] == True:
            if self.delta_ratio == 0:
                if shared_infer == False:
                    up = self.Wmean3(x)
                    gate = self.Wmean1(x)
                    return self.Wmean2(self.act_fn(gate) * up)
                else:
                    return self.Wmean2(self.act_fn(current_gate_hidden_states) * current_up_hidden_states)

            if shared_infer == False:
                up = self.Wmean3(x) + self.delta_u3(self.delta_v3(x))
                gate = self.Wmean1(x) + self.delta_u1(self.delta_v1(x))
                down_proj = self.Wmean2(self.act_fn(gate) * up) + self.delta_u2(self.delta_v2(self.act_fn(gate) * up))
            else:
                if self.delta_share_V == True:
                    up = current_up_hidden_states + self.delta_u3(current_up_deltav_hidden_states)
                    gate = current_gate_hidden_states + self.delta_u1(current_gate_deltav_hidden_states)
                    return self.Wmean2(self.act_fn(gate) * up) + self.delta_u2(self.delta_v2(self.act_fn(gate) * up))
                else:
                    up = current_up_hidden_states + self.delta_u3(self.delta_v3(x))
                    gate = current_gate_hidden_states + self.delta_u1(self.delta_v1(x))
                    return self.Wmean2(self.act_fn(gate) * up) + self.delta_u2(self.delta_v2(self.act_fn(gate) * up))

            return down_proj
        elif cfg['no_probe_process'] == True:
            if self.delta_ratio == 0:
                if shared_infer == False:
                    up = self.Wmean3(x)
                    gate = self.Wmean1(x)
                    return self.Wmean2(self.act_fn(gate) * up)
                else:
                    return current_down_hidden_states
            if self.layer_order not in cfg['skip_layers']:
                if 'probe' in cfg['prune_method']:
                    probe_out_dim_indices = None
                    if cfg['mode'] == 'sync':
                        # probe_out_dim_indices, probe_out = self.probe_process(x, **kwargs)
                        keep_dim = int(self.hidden_dim * (1 - cfg['prune_ratio']))
                        probe_out_dim_indices = torch.arange(keep_dim)
                        # count flops for probe
                        if cfg['onlyprobe'] == True:
                            # match the shape, and will not count the flops for this part
                            down_proj = torch.zeros((cfg['batch_size'], x.shape[1], self.hidden_size), device=x.device, dtype=x.dtype)
                            return down_proj
                    elif cfg['mode'] == 'asyncintra':
                        if 'post_layernorm_attn_residual' in kwargs:
                            # _, _ = self.probe_process(kwargs['post_layernorm_attn_residual'], **kwargs)
                            pass
                        else:
                            pass

                    if 'recordcommonchannel' in cfg['prune_method']:
                        self.mlp_cur_select_indices = probe_out_dim_indices.tolist()

                    if self.delta_ratio == 0:
                        if shared_infer == False:
                            up = self.Wmean3(x)
                            gate = self.Wmean1(x)
                            return self.Wmean2(self.act_fn(gate) * up)
                        else:
                            return current_down_hidden_states
                    if shared_infer == False:
                        up = self.Wmean3(x, out_dim_indices=probe_out_dim_indices) + self.delta_u3(self.delta_v3(x))[:, probe_out_dim_indices]
                        gate = self.Wmean1(x, out_dim_indices=probe_out_dim_indices) + self.delta_u1(self.delta_v1(x))[:, probe_out_dim_indices]
                        main_down_proj = self.Wmean2(self.act_fn(gate) * up, in_dim_indices=probe_out_dim_indices)
                        delta_down_proj = self.delta_u2(F.linear( (self.act_fn(gate) * up), self.delta_v2.weight[:, probe_out_dim_indices], bias=None))
                        down_proj = main_down_proj + delta_down_proj
                    else:
                        if self.delta_share_V == True:
                            up = current_up_hidden_states + self.delta_u3(current_up_deltav_hidden_states)[:, probe_out_dim_indices]
                            gate = current_gate_hidden_states + self.delta_u1(current_gate_deltav_hidden_states)[:, probe_out_dim_indices]
                            return current_down_hidden_states + self.delta_u2(current_down_deltav_hidden_states)
        elif cfg['calibration_stage'] == True:
            up = self.Wmean3(x) + self.delta_u3(self.delta_v3(x))
            gate = self.Wmean1(x) + self.delta_u1(self.delta_v1(x))
            down_proj = self.Wmean2(self.act_fn(gate) * up) + self.delta_u2(self.delta_v2(self.act_fn(gate) * up))
            return down_proj
        elif cfg['calibration_stage'] == False:
            if self.layer_order not in cfg['skip_layers']:
                if 'probe' in cfg['prune_method']:
                    probe_out_dim_indices = None
                    if cfg['mode'] == 'sync':
                        probe_out_dim_indices, probe_out = self.probe_process(x, **kwargs)
                        # count flops for probe
                        if cfg['onlyprobe'] == True:
                            # match the shape, and will not count the flops for this part
                            down_proj = torch.zeros((cfg['batch_size'], x.shape[1], self.hidden_size), device=x.device, dtype=x.dtype)
                            return down_proj
                    elif cfg['mode'] == 'asyncintra':
                        if 'post_layernorm_attn_residual' in kwargs:
                            _, _ = self.probe_process(kwargs['post_layernorm_attn_residual'], **kwargs)
                        else:
                            pass

                    if 'recordcommonchannel' in cfg['prune_method']:
                        self.mlp_cur_select_indices = probe_out_dim_indices.tolist()

                    up = self.Wmean3(x, out_dim_indices=probe_out_dim_indices) + self.delta_u3(self.delta_v3(x))[:, probe_out_dim_indices]
                    gate = self.Wmean1(x, out_dim_indices=probe_out_dim_indices) + self.delta_u1(self.delta_v1(x))[:, probe_out_dim_indices]
                    main_down_proj = self.Wmean2(self.act_fn(gate) * up, in_dim_indices=probe_out_dim_indices)
                    delta_down_proj = self.delta_u2(F.linear( (self.act_fn(gate) * up), self.delta_v2.weight[:, probe_out_dim_indices], bias=None))
                    down_proj = main_down_proj + delta_down_proj
                    return down_proj
            else:
                up = self.Wmean3(x) + self.delta_u3(self.delta_v3(x))
                gate = self.Wmean1(x) + self.delta_u1(self.delta_v1(x))
                down_proj = self.Wmean2(self.act_fn(gate) * up) + self.delta_u2(self.delta_v2(self.act_fn(gate) * up))
                return down_proj