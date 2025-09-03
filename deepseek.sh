# # get expert frequencies in advance, save a json file
# python preprocess/get_expert_freq.py \
#     --base_model_path=/home/tom/fsas/models/deepseek-ai/deepseek-moe-16b-base \
#     --save_path=/home/tom/fsas/zhanghongyu/results/deepseek \
#     --model_type=deepseek \
#     --dataset_name=wikitext \
#     --split=train \
#     --seed=42 \
#     --max_samples=20000 \

# # get fisher information of model in advance
# python preprocess/get_fisher.py \
#     --base_model_path=/home/tom/fsas/models/deepseek-ai/deepseek-moe-16b-base \
#     --save_path=/home/tom/fsas/zhanghongyu/results/deepseek \
#     --num_samples=1024 \
# 	--scale_type fisher \

# # get the SVD scale of model in advance
# python preprocess/get_scale.py \
#     --base_model_path=/home/tom/fsas/models/deepseek-ai/deepseek-moe-16b-base \
#     --save_path=/home/tom/fsas/zhanghongyu/results/deepseek \
#     --model_type=deepseek \
#     --dataset_name=wikitext \
#     --split=train \
#     --seed=42 \
#     --max_samples=256 \

# run the D2-MoE, the evaluation results will be saved in the result_path
python D2-deepseek.py \
    --control_name=wikitext-2v1_llama-2-7b_clm_20_1024_0.1_ppwandasp_probe-default_sync_c4-2000_0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank_default \
    --base_model_path=/home/tom/fsas/models/deepseek-ai/deepseek-moe-16b-base \
    --expert_freq_path=/home/tom/fsas/zhanghongyu/results/deepseek/deepseek_wikitext_20000_expert_frequencies.json \
    --fisher_path=/home/tom/fsas/zhanghongyu/results/deepseek/fisher_deepseek-moe-16b-base_processed.pt \
    --svd_scale_path=/home/tom/fsas/zhanghongyu/results/deepseek/SVD_scale_deepseek_all_256.pt \
    --result_path=/home/tom/fsas/zhanghongyu/results/deepseek/ \
    --pp_ratio=0.2 \
    --delta_ratio=0.8 \
    --share_ratio=1 \
    --merge_method=fisher \