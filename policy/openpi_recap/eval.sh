#!/bin/bash

policy_name=openpi_recap
task_name=${1}
task_config=${2}
ckpt_setting=${3:-remote}
server_host=${4:-127.0.0.1}
server_port=${5:-12345}
seed=${6:-0}

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --server_host ${server_host} \
    --server_port ${server_port} \
    --seed ${seed} \
    --policy_name ${policy_name}
