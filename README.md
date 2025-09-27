# Streaming Attention Approximation via Discrepancy Theory

This repository contains code for evaluating the proposed BalanceKV algorithm on the LongBench benchmark. 

### Files and Directories

- **src/balanced_walk.py**: Implements the balanced walk algorithm.
- **src/llama_forward.py**: Contains functions for inference using the Llama model and patching our custom KV cache.
- **src/metrics_longbench.py**: Defines various metrics used for LongBench evaluation.
- **src/single_layer_approx.ipynb**: Notebook containing ablation study experiments regarding single layer attention approximation.
- **run_longbench.py**: Main script for running the LongBench evaluation.
- **run_needle_in_haystack.py**: Main script for running the NIAH evaluation.

## Requirements

Install requirements via
```sh
pip install -r requirements.txt
```

## Usage

To run the LongBench evaluation, use the following command (default model is ``meta-llama/Llama-3.1-8B-Instruct'')

```sh
python run_longbench.py --kv_type weightedbw --datasets qasper --e
```

To run the NIAH evaluation, use the following command (default model is ``meta-llama/Llama-3.1-8B-Instruct'')
```sh
python run_needle_in_haystack.py --kv_type "weightedbw" --haystack_dir "<CurrentPath>/data/PaulGrahamEssays"
```