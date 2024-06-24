base_path=${1-"."}
ckpt=${2-"."}
port=2040


for data in dolly self_inst vicuna sinst uinst 
do
    for seed in 10 20 30 40 50
    do
        bash ${base_path}/scripts/llama/eval/eval_main_${data}.sh ${base_path} ${port} 1 ${ckpt} --seed $seed  --eval-batch-size 8
    done
done
