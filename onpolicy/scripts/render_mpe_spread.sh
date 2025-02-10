#!/bin/sh
env="MPE"
scenario="simple_spread"
num_landmarks=3
num_agents=3
algo="mappo"
exp="simple_spread"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed 3 \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 25 --render_episodes 1 \
    --model_dir "/home/thales/jszpirer/sugar/mappo/onpolicy/scripts/results/MPE/simple_spread/mappo/simple_spread/wandb/latest-run/files" \
    --use_wandb False
done
