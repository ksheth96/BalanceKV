
LONGBENCH_V1_E=("qasper" "multifieldqa_en" "hotpotqa" "2wikimqa" \
    "gov_report" "multi_news" "trec" "triviaqa" "samsum" \
    "passage_count" "passage_retrieval_en" "lcc" "repobench-p")

METHODS=("exact" "snapkv" "snapkv2" "balancekv")

for METHOD in "${METHODS[@]}"; do
    for dataset in ${LONGBENCH_V1_E[@]}; do
        dataset="${dataset}_e"
        # echo "Running for dataset: ${dataset}"
        cmd="python run.py --method $METHOD --dataset longbench-e --datadir $dataset --compression_ratio 0.75"
        echo $cmd
        eval $cmd
    done
done
