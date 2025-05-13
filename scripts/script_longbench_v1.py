import os, sys
import argparse

LONGBENCH_V1_E=["qasper", "multifieldqa_en","hotpotqa","2wikimqa", "gov_report", "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

# model list
# - meta-llama/Llama-3.1-8B-Instruct
# - Qwen/Qwen2.5-7B-Instruct
# - Qwen/Qwen2.5-14B-Instruct
# - Qwen/Qwen2.5-32B-Instruct
# - meta-llama/Llama-3.2-3B-Instruct
# - Gensyn/Qwen2.5-0.5B-Instruct
# - Qwen/Qwen2.5-1.5B-Instruct
# - Qwen/Qwen2.5-3B-Instruct
# - Qwen/Qwen2.5-0.5B-Instruct
# - mistralai/Mistral-7B-Instruct-v0.3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument('--kv_type', type=str, default="exact")
    args = parser.parse_args()

    cmd_all = []

    for dataset in LONGBENCH_V1_E:
        cmd_all.append(f'python run_longbench_v1.py --e --model_name {args.model_name}  --kv_type {args.kv_type} --dataset {dataset}')

    for cmd in cmd_all:
        print(cmd)
        os.system(cmd)
    

if __name__ == "__main__":
    main()