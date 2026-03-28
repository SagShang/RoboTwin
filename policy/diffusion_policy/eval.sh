#!/bin/bash

policy_name=diffusion_policy
task_name=${1}
task_config=${2}
ckpt_setting=${3}
server_host=${4:-172.18.1.203}
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
    --seed ${seed}
