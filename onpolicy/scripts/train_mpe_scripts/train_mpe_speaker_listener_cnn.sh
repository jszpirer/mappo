#!/bin/sh
env="MPE"
scenario="simple_speaker_listener_cnn_sparse_local"
num_landmarks=3
num_agents=2
grid_resolution=77
nb_additional_data=2
out_channels=3
output_comm=3
stride=2
kernel=7
obs_range=12
algo="mappo" #"rmappo" "ippo"
exp="simple_speaker_listener_cnn_sparse77_local"
seed_max=1
project="simple_speaker_listener_normal"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
# for seed in `seq ${seed_max}`;
for seed in 1 2 3 4 5;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 125 --num_env_steps 15000000 \
    --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --user_name "jeanne-szpirer-universit-libre-de-bruxelles" --project_name ${project} \
    --grid_resolution ${grid_resolution} --nb_additional_data ${nb_additional_data} --share_policy --num_output_channels ${out_channels} --output_comm ${output_comm} \
    --stride ${stride} --kernel ${kernel} --use_feature_normalization --recurrent_N 1 --obs_range ${obs_range}\
    #--curriculum_start 900000
    #--model_dir "/home/thales/jszpirer/sugar/mappo/onpolicy/scripts/results/MPE/simple_speaker_listener_cnn_local_curriculum/rmappo/simple_speaker_listener_cnn_local_12/wandb/run-20250409_133451-jginmd95/files"
done
