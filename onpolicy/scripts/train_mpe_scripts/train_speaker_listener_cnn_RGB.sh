#!/bin/sh
env="MPE"
scenario="simple_speaker_listener_cnn_goalcolor_step2"
num_landmarks=3
num_agents=2
grid_resolution=32
nb_additional_data=2
out_channels=3
output_comm=3
stride_comm=1
kernel_comm=3
simpleCNN=0
simpleCNN2=0
simpleCNN3=0
simpleCNN4=1
stride=1
kernel=3
algo="mappo" #"rmappo" "ippo"
exp="simple_speaker_listener_cnn_goalcolor_step232kernel3reluabsscale"
seed_max=10
project="simple_speaker_listener_steps"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
#for seed in `seq ${seed_max}`;
for seed in 1 2 3 4 5
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 125 --num_env_steps 15000000 \
    --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --user_name "jeanne-szpirer-universit-libre-de-bruxelles" --project_name ${project} \
    --grid_resolution ${grid_resolution} --nb_additional_data ${nb_additional_data} --share_policy --num_output_channels ${out_channels} --output_comm ${output_comm} \
    --stride_comm ${stride_comm} --simpleCNN ${simpleCNN} --kernel_comm ${kernel_comm} --stride ${stride} --kernel ${kernel} --simpleCNN2 ${simpleCNN2} \
    --simpleCNN3 ${simpleCNN3} --simpleCNN4 ${simpleCNN4}
done
