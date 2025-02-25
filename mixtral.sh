# get expert frequencies in advance, save a json file
python preprocess/get_expert_freq.py \
    --base_model_path=your_model_path \
    --save_path=your_save_path \
    --model_type=mixtral \
    --dataset_name=wikitext \
    --split=train \
    --seed=42 \
    --max_samples=20000 \

# get fisher information of model in advance
python preprocess/get_fisher.py \
    --base_model_path=your_model_path \
    --save_path=your_save_path \
    --num_samples=1024 \
	--scale_type fisher \

# get the SVD scale of model in advance
python preprocess/get_scale.py \
    --base_model_path=your_model_path \
    --save_path=your_save_path \
    --model_type=mixtral \
    --dataset_name=wikitext \
    --split=train \
    --seed=42 \
    --max_samples=256 \

# run the D2-MoE, the evaluation results will be saved in the result_path
python D2-mixtral.py \
    --control_name=wikitext-2v1_llama-2-7b_clm_20_1024_0.1_ppwandasp_probe-default_sync_c4-2000_0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-0.5+0.05-seqrank+bszrank_default \
    --base_model_path=your_mixtral_model_path \
    --expert_freq_path=your_expert_freq_path \
    --fisher_path=your_fisher_path \
    --svd_scale_path=your_svd_scale_path \
    --result_path=your_result_path \
    --pp_ratio=0.2 \
    --delta_ratio=0.8 \
    --share_ratio=1 \
    --merge_method=fisher \