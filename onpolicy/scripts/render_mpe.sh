#!/bin/sh
env="MPE"
scenario="aggregation_1_mappo"
num_landmarks=2
num_agents=5
algo="mappo"
exp="aggregation_1"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed 5 \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 125 --render_episodes 1 \
    --model_dir "/home/jszpirer/RL/mappo/onpolicy/scripts/results/MPE/aggregation_1_mappo/mappo/aggregation_1/wandb/run-20250122_200606-6bdoenf1/files" \
    --use_wandb False
done
