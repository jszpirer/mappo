#!/bin/sh
env="MPE"
scenario="simple_reference_cnn_sparse_local"
num_landmarks=3
num_agents=2
grid_resolution=77
nb_additional_data=2
out_channels=3
output_comm=10
kernel=7
stride=2
algo="mappo"
exp="simple_reference_cnn_sparse77_local"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    random_seed=$(date +%s%N | cut -b10-19)
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${random_seed} \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 125 --render_episodes 4 \
    --model_dir "/home/thales/jszpirer/sugar/mappo/onpolicy/scripts/results/MPE/simple_reference_cnn_sparse_local/mappo/simple_reference_cnn_sparse77_local/wandb/run-20250513_171214-yzo9n08y/files" \
    --use_wandb False --grid_resolution ${grid_resolution} --nb_additional_data ${nb_additional_data} --stride ${stride} \
    --kernel ${kernel} --num_output_channels ${out_channels} --output_comm ${output_comm}
done
