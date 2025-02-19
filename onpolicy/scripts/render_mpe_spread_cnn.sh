#!/bin/sh
env="MPE"
scenario="simple_spread_cnn_Maurolocal_initpos"
num_landmarks=3
num_agents=3
grid_resolution=32
nb_additional_data=1
algo="mappo"
exp="simple_spread_cnn_Maurolocal_initpos"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed 2 \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 125 --render_episodes 4 \
    --model_dir "/home/thales/jszpirer/sugar/mappo/onpolicy/scripts/results/MPE/simple_spread_cnn_Maurolocal_initpos/mappo/simple_spread_cnn_Maurolocal_initpos/32x32_3.85agents/files" \
    --use_wandb False --grid_resolution ${grid_resolution} --nb_additional_data ${nb_additional_data}
done
