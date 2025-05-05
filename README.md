# BalanceKV: KV Cache Compression through Discrepancy Theory

This repository contains code for evaluating the proposed BalanceKV algorithm on the LongBench benchmark. 

Implementations are based on on https://github.com/NVIDIA/kvpress.git which supports various evaluations:

- RULER (https://github.com/NVIDIA/RULER)
- LongBench / LongBench-e (https://github.com/THUDM/LongBench/tree/main/LongBench)
- LongBench-v2 (https://github.com/THUDM/LongBench)
- Infinite-Bench (https://github.com/OpenBMB/InfiniteBench)
- Loogle (https://github.com/bigai-nlco/LooGLE)

## Files and Directories

- run.py: Main script for running the LongBench evaluation.
- balancekv_press.py: Implements the balanced walk algorithm.

## Usage

To run LongBench experiment, 
```python
python run.py --method balancekv --dataset longbench-e --datadir qasper
```

To run RULER experiment, 
```python
python run.py --method balancekv --dataset ruler --datadir 4096
```

