#!/bin/sh
env="MPE"
scenario="simple_spread_cnn_sparse_3.85agentslandmarks_local" 
num_landmarks=10
num_agents=10
grid_resolution=77
nb_additional_data=2
noise=0
stride=2
kernel=7
algo="rmappo" #"mappo" "ippo"
exp="simple_spread_cnn_sparse77_rnn_local"
seed_max=5
project="simple_spread_10agents"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
# for seed in `seq ${seed_max}`;
for seed in 5
do
    echo "seed ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 125 --num_env_steps 100000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --user_name "jeanne-szpirer-universit-libre-de-bruxelles" --project_name ${project} \
    --grid_resolution ${grid_resolution} --nb_additional_data ${nb_additional_data} --wheel_noise ${noise} \
    --stride ${stride} --kernel ${kernel} --save_interval 1000000
done
