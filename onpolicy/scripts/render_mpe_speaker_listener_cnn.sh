#!/bin/sh
env="MPE"
scenario="simple_speaker_listener_cnn"
num_landmarks=3
num_agents=2
grid_resolution=32
nb_additional_data=2
out_channels=3
kernel=3
stride=1
algo="mappo"
exp="simple_speaker_listener_cnn"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    random_seed=$(date +%s%N | cut -b10-19)
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${random_seed} \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 250 --render_episodes 4 \
    --model_dir "/home/thales/jszpirer/sugar/mappo/onpolicy/scripts/results/MPE/simple_speaker_listener_cnn/mappo/simple_speaker_listener_cnn/wandb/run-20250325_180755-n6mdmixj/files" \
    --use_wandb False --grid_resolution ${grid_resolution} --nb_additional_data ${nb_additional_data} --share_policy --stride ${stride} \
    --kernel ${kernel} --num_output_channels ${out_channels} --use_feature_normalization
done
