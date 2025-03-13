#!/bin/sh
env="MPE"
scenario="simple_speaker_listener_cnn_goalcolor"
num_landmarks=3
num_agents=2
grid_resolution=48
nb_additional_data=2
out_channels=3
output_comm=3
stride_comm=8
stride=2
kernel_comm=8
sigmoid=1
algo="mappo" #"rmappo" "ippo"
exp="simple_speaker_listener_cnn_goalcolorsharedcriticsigmoidkernel8"
seed_max=5
project="simple_speaker_listener_noise"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 125 --num_env_steps 15000000 \
    --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --user_name "jeanne-szpirer-universit-libre-de-bruxelles" --project_name ${project} \
    --grid_resolution ${grid_resolution} --nb_additional_data ${nb_additional_data} --share_policy --num_output_channels ${out_channels} --output_comm ${output_comm} \
    --stride ${stride} --sigmoid ${sigmoid} --stride_comm ${stride_comm} --kernel_comm ${kernel_comm}
done
