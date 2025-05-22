#!/bin/sh
env="MPE"
scenario="aggregation_cnn_sparse_3.85agentslandmarks_local"
num_landmarks=1
num_agents=7
grid_resolution=77
stride=2
kernel=7
nb_additional_data=2
algo="mappo"
exp="aggregation_cnn_sparse77_local"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    random_seed=$(date +%s%N | cut -b10-19)
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${random_seed} \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 125 --render_episodes 4 \
    --model_dir "/home/thales/jszpirer/sugar/mappo/onpolicy/scripts/results/MPE/aggregation_cnn_sparse_3.85agentslandmarks_local/mappo/aggregation_cnn_sparse77_local/wandb/run-20250514_154051-5t6jjto2/files" \
    --use_wandb False --grid_resolution ${grid_resolution} --nb_additional_data ${nb_additional_data} \
    --stride ${stride} --kernel ${kernel}
done
