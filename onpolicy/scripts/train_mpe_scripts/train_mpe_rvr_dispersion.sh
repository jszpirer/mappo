#!/bin/sh
env="MPE"
scenario="rvr_dispersion_local_omni_walls_fast" 
num_landmarks=0
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
project="rvr_dispersion_walls_tests"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
# for seed in `seq ${seed_max}`;
for seed in 2
do
    echo "seed is ${seed}:"
    TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python -m cProfile -o prof.cprof ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 125 --num_env_steps 100000\
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --user_name "jeanne-szpirer-universit-libre-de-bruxelles" --project_name ${project} \
    --grid_resolution ${grid_resolution} --nb_additional_data ${nb_additional_data} --wheel_noise ${noise} \
    --stride ${stride} --kernel ${kernel} --save_interval 1000000 --grid_resolution_critic ${grid_resolution_critic} --omniscient_critic --use_directions --dim_actor 4 --padding ${padding}
done
