#!/bin/sh
env="MPE"
scenario="simple_coverage_cnn_sparse" 
num_landmarks=0
num_agents=5
grid_resolution=77
nb_additional_data=2
noise=0
stride=2
kernel=7
<<<<<<< HEAD
algo="rmappo" #"mappo" "ippo"
exp="simple_coverage_cnn_sparse77_local"
seed_max=5
project="simple_coverage_10agents"
=======
algo="rmappo" #"rmappo" "ippo"
exp="simple_coverage_cnn_sparse77_rnn_testbatch2"
seed_max=5
project="simple_coverage_5agents"
>>>>>>> 373351a6676aeb2e1b6faf137a8e5c7b3aaa15f3

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
# for seed in `seq ${seed_max}`;
for seed in 1
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
<<<<<<< HEAD
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 125 --num_env_steps 50000000 \
=======
    --n_training_threads 1 --n_rollout_threads 128 --batch_size 128 --num_mini_batch 1 --episode_length 125 --num_env_steps 50000000 \
>>>>>>> 373351a6676aeb2e1b6faf137a8e5c7b3aaa15f3
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --user_name "jeanne-szpirer-universit-libre-de-bruxelles" --project_name ${project} \
    --grid_resolution ${grid_resolution} --nb_additional_data ${nb_additional_data} --wheel_noise ${noise} \
    --stride ${stride} --kernel ${kernel} --save_interval 1000000
done
