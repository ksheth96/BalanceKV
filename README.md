# BalanceKV: KV Cache Compression through Discrepancy Theory

This repository contains code for evaluating the proposed BalanceKV algorithm on the LongBench benchmark. 

### Files and Directories

- **balanced_walk.py**: Implements the balanced walk algorithm.
- **llama_simple.py**: Contains functions for inference the LLaMA model.
- **metrics_longbench.py**: Defines various metrics used for LongBench evaluation.
- **run_longbench_v1.py**: Main script for running the LongBench evaluation.

## Usage

To run the LongBench evaluation, use the following command (default model is ``meta-llama/Llama-3.1-8B-Instruct'')

```sh
python run_longbench_v1.py --kv_type weighedbw --datasets qasper --e
```

## Requirement

- PyTorch (tested on 2.5.1)
- MInference >= 0.1.5.post1
- Transformers >= 4.47.1
- Datasets
- NumPy

We will add the code for single layer attention approximation in the next commit.