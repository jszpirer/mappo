#!/bin/sh
env="MPE"
scenario="rvr_spread_local_omni_walls_fast_cells"
num_landmarks=4
num_agents=5
grid_resolution=81
grid_resolution_critic=81
nb_additional_data=2
noise=0
stride=2
kernel=9
padding=0
algo="rmappo" #"mappo" "ippo"
exp="rvr_local_omni_12and5both"
seed_max=10
project="rvr_dispersion_walls"

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    random_seed=$(date +%s%N | cut -b10-19)
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${random_seed} \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 125 --render_episodes 4 \
    --model_dir "/home/thales/jszpirer/sugar/mappo/onpolicy/scripts/results/rvrspreadwallscells5agents_seed1/files" \
    --use_wandb False --grid_resolution ${grid_resolution} --nb_additional_data ${nb_additional_data} \
    --stride ${stride} --kernel ${kernel} --grid_resolution_critic ${grid_resolution_critic} --omniscient_critic --use_directions --dim_actor 4 --padding ${padding}
done
