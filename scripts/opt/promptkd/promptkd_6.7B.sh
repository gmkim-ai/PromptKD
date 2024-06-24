#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-4}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"."}
CKPT_NAME="6.7B-init"
CKPT="${BASE_PATH}/checkpoints/opt/train/minillm_init/opt-6.7B"
# CKPT="facebook/opt-6.7b" # download automatically
TEACHER_CKPT_NAME="13B-sft"
TEACHER_CKPT="${BASE_PATH}/checkpoints/opt/train/sft/opt-13B/"
# MP_SIZE=4
# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/opt/"
# hp
BATCH_SIZE=16
LR=0.000005
GRAD_ACC=1
EVAL_BATCH_SIZE=16
# prompt
TEACHER_LR=0.000005
PROMPT_LEN=7
PROMPT_INIT_TEXT="Suppose you are a student."
KLD_TYPE="reverse"
BASE_KLD_TYPE="reverse"
BASE_COEF=1.0
# length
MAX_LENGTH=512
# runtime
SAVE_PATH="${BASE_PATH}/results/opt/train/promptkd/opt-6.7B-13B"
# seed
SEED=10


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type opt"
OPTS+=" --gradient-checkpointing"
# OPTS+=" --model-parallel"
# OPTS+=" --model-parallel-size ${MP_SIZE}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 4"
OPTS+=" --dev-num 1000"
# prompt
OPTS+=" --teacher-peft prompt_init"
OPTS+=" --teacher-lr ${TEACHER_LR}"
OPTS+=" --prompt-len ${PROMPT_LEN}"
OPTS+=" --prompt-init-text ${PROMPT_INIT_TEXT}"
OPTS+=" --base-coef ${BASE_COEF}"
OPTS+=" --teacher-kld-type ${KLD_TYPE}"
OPTS+=" --base-kld-type ${BASE_KLD_TYPE}"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 100"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 10"
OPTS+=" --kd-ratio 0.5"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 10"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero2_offload.json"
# type
OPTS+=" --type promptkd"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/train_promptkd.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
