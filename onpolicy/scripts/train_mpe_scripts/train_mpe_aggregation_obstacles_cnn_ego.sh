#!/bin/sh
env="MPE"
scenario="simple_aggregation_obs_cnn_ego" 
num_landmarks=0
num_agents=5
nb_additional_data=2
grid_resolution=77
noise=0
algo="rmappo" #"mappo" "ippo"
exp="local_coverage_aggregation_obs_cnn_ego"
seed_max=5
project="epuck_aggregation_5agents_obs_test"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
# for seed in `seq ${seed_max}`;
for seed in 1
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 125 --num_env_steps 50000000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-5 --critic_lr 7e-4 --user_name "jeanne-szpirer-universit-libre-de-bruxelles" --project_name ${project} \
    --grid_resolution ${grid_resolution} --nb_additional_data ${nb_additional_data} --grid_resolution_critic 77 --num_obstacles 4 \
    --save_interval 1000000 --omniscient_critic --velocities_critic --kernel 7 --stride 2 --use_directions
done
