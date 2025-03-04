#!/bin/sh
env="MPE"
scenario="simple_reference_cnn_local"
num_landmarks=3
num_agents=2
grid_resolution=32
nb_additional_data=2
algo="mappo"
exp="simple_reference_cnn_local"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed 1 \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 125 --render_episodes 4 \
    --model_dir "/home/thales/jszpirer/sugar/mappo/onpolicy/scripts/results/MPE/simple_reference_cnn_local/mappo/simple_reference_cnn_local/wandb/3.85agents_seed1/files" \
    --use_wandb False --grid_resolution ${grid_resolution} --nb_additional_data ${nb_additional_data}
done
