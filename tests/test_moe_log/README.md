# Where to Hook
- `vllm/model_executor/layers/fused_moe/moe_log.py` defines the loger class `MoETopKLogger` and instance `MOE_TOPK_LOGGER`
- Place hook at `vllm/model_executor/layers/fused_moe/layer.py` 
    - log `layer_idx, topk_ids, topk_weights,top_k` after `topk_weights, topk_ids = FusedMoE.select_experts` in `UnquantizedFusedMoEMethod.forward_cuda`
        - `layer_idx` is dynamically added to `UnquantizedFusedMoEMethod` in `FusedMoE.__init__` by `self.quant_method.layer_idx = layer_idx`. `layer_idx` is evaluated from arg `prefix` passed to `FusedMoE.__init__`
- vllm/model_executor/layers/fused_moe/moe_log_context.py deals with `req_id` from `vllm/worker/model_runner.py`. 
    - Records request ids for all tokens at `token_req_ids` in `ModelInputForGPUBuilder.build` 
    - Pass (and store in thread local) request ids and clear (in thread local) at `ModelRunner.execute_model`
    - `MoETopKLogger.log_topk` at `vllm/model_executor/layers/fused_moe/moe_log.py` receives request ids.
- Environment variables are controlled in `vllm/envs.py`
    - `$VLLM_LOG_MOE` sets the path where log is saved in .json fomat
    - `$LOG_MOE_LAYER_IDX` indicates the layer to be logged by layer_idx
    - `$LOG_MOE_SEED` is for reproductivity of the experiment scripted by `tests/test_moe_log/run_generate.py`

# Results
Sorry that I don't have appropriate server to run vllm because I only have the server with Ascend. Unfortunately, the server for running vllm on V100 in my previous project (requests scheduling system for LLM inference ) is not available now.

I don't know Top-3 experts. But We compute the empirical routing distribution over experts as

$$
p_i = \frac{c_i}{\sum_j c_j},
$$

where $c_i$ is the number of times expert $i$ is selected.
The routing entropy is defined as

$$
H = - \sum_i p_i \log p_i,
$$

and we report the normalized entropy $H / \log N$ for comparability across models.

Entropy measures how evenly tokens are distributed across experts: lower entropy indicates routing concentration, while higher entropy suggests balanced expert utilization.

# Command to Run
- generate `prompts.txt` by `cd path/to/vllm && python tests/test_moe_log/make_prompts.py`
- run the LLM inference without log by
    - `cd path/to/vllm`
    - `unset VLLM_LOG_MOE`
    - `python tests/test_moe_log/run_generate.py`
        - the seed is automatically set to default `1234` if not set
- run the LLM inference with log by
    - `cd path/to/vllm`
    - `VLLM_LOG_MOE = path/to/log.json LOG_MOE_LAYER_IDX = <layer_idx> python tests/test_moe_log/run_generate.py`
- run plotting by `plot_hist.py` (please set `$VLLM_LOG_MOE` as path of log.json to read).