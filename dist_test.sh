CONFIG=$1

python eval.py --config $CONFIG --work-dir ./work_logs --sam_refine

#not support distributed inference yet
#WORK_DIR=${WORK_DIR:-"./work_logs"}
#GPUS=${GPUS:-1}
#NNODES=${NNODES:-1}
#NODE_RANK=${NODE_RANK:-0}
#PORT=${PORT:-29501}
#MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
#
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python -m torch.distributed.launch \
#    --nnodes=$NNODES \
#    --node_rank=$NODE_RANK \
#    --master_addr=$MASTER_ADDR \
#    --nproc_per_node=$GPUS \
#    --master_port=$PORT \
#    $(dirname "$0")/eval.py \
#    --config $CONFIG \
#    --work-dir $WORK_DIR \
#    --launcher pytorch \
#    ${@:4}