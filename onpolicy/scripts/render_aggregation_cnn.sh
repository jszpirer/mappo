#!/bin/sh
env="MPE"
scenario="simple_coverage_aggregation_cnn_sparse_localj_noise_critic_action"
num_landmarks=0
num_agents=10
grid_resolution=77
stride=2
kernel=7
nb_additional_data=2
algo="rmappo"
exp="local_aggregation_coverage_afterICRA"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    random_seed=$(date +%s%N | cut -b10-19)
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${random_seed} \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 125 --render_episodes 1 \
    --model_dir "/home/thales/jszpirer/sugar/mappo/onpolicy/scripts/results/aggregation10agentslocalnoisecurrisulum_seed6/files" \
    --use_wandb False --grid_resolution ${grid_resolution} --nb_additional_data ${nb_additional_data} --grid_resolution_critic 77 \
    --stride ${stride} --kernel ${kernel} --omniscient_critic --use_directions
done
