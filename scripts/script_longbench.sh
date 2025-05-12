
LONGBENCH_V1_E=("qasper" "multifieldqa_en" "hotpotqa" "2wikimqa" \
    "gov_report" "multi_news" "trec" "triviaqa" "samsum" \
    "passage_count" "passage_retrieval_en" "lcc" "repobench-p")

# METHODS=("exact" "snapkv" "snapkv2" "balancekv")
METHOD=$1
# MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
# for METHOD in "${METHODS[@]}"; do
for dataset in ${LONGBENCH_V1_E[@]}; do
    # echo "Running for dataset: ${dataset}"
    cmd="python run.py --method $METHOD --dataset longbench-e --datadir $dataset --compression_ratio 0.75 --model $MODEL"
    echo $cmd
    eval $cmd
done
# done

# python run.py --method exact --dataset longbench-e --datadir qasper --compression_ratio 0.75 --model Qwen/Qwen2.5-14B-Instruct
