
SEQLENS=("4096" "8192" "16384")

# METHODS=("exact" "snapkv" "snapkv2" "balancekv")
if [ $# -eq 0 ]; then
    echo "Usage: $0 <method>"
    exit 1
fi
METHODS=($1)
for METHOD in "${METHODS[@]}"; do
    for dataset in ${SEQLENS[@]}; do
        # echo "Running for dataset: ${dataset}"
        # cmd="python run.py --method $METHOD --dataset ruler --datadir $dataset --compression_ratio 0.75"
        cmd="python run.py --method $METHOD --dataset ruler --datadir $dataset"
        echo $cmd
        eval $cmd
    done
done
