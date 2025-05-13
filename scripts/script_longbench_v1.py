import os, sys
import argparse

LONGBENCH_V1_E=["qasper", "multifieldqa_en","hotpotqa","2wikimqa", "gov_report", "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

#                        | qasper     multa_en   hotpotqa   2wikimqa   gov_port   multnews   trec       triviaqa   samsum     passount   passl_en   lcc        repoch-p    | average
# exact                   | 0.4456     0.5065     0.6914     0.6039     -2.0       -1.0000    0.7533     -1.0000    0.4317     0.2200     0.9967     -1.0000    -1.0000     | -1.0000    | exact
# pyramidkv               | 0.3447     0.4633     0.6778     0.5592     0.1516     0.1539     0.6933     0.6324     0.4036     0.2267     0.9933     -1.0000    -1.0000     | -1.0000    | pyramidkv
# snapkv                  | 0.3621     0.4678     0.6664     0.5702     0.1635     0.1607     0.7033     -1.0000    0.4108     0.2200     0.9933     -1.0000    -1.0000     | -1.0000    | snapkv
# streamingllm            | 0.2012     -2.0       -2.0       -2.0       -2.0       -2.0       -2.0       -2.0       -2.0       -2.0       -2.0       -2.0       -2.0        | -1.0000    | streamingllm
# sink                    | 0.2967     0.3147     -2.0       0.5222     0.2083     0.1821     0.6800     -1.0000    0.4241     0.2133     0.7456     -1.0000    -1.0000     | -1.0000    | sink
# weightedbw2             | 0.4014     0.4317     0.6446     0.5806     0.2226     0.2032     0.7300     -1.0000    0.4107     -1.0000    0.9200     -1.0000    -1.0000     | -1.0000    | weightedbw2

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
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-3B-Instruct")
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