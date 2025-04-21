
SEQLENS=("4096" "8192" "1684")

METHODS=("exact" "snapkv" "snapkv2" "balancekv")

for METHOD in "${METHODS[@]}"; do
    for dataset in ${SEQLENS[@]}; do
        # echo "Running for dataset: ${dataset}"
        cmd="python run.py --method $METHOD --dataset ruler --datadir $dataset --compression_ratio 0.75"
        echo $cmd
        eval $cmd
    done
done
