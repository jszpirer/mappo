#!/bin/sh
env="MPE"
scenario="speaker_listener_multiple"
num_landmarks=6
num_agents=3
noise=0
algo="mappo"
exp="simple_speaker_listener"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    random_seed=$(date +%s%N | cut -b10-19)
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${random_seed} \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 125 --render_episodes 4 \
    --model_dir "/home/thales/jszpirer/sugar/mappo/onpolicy/scripts/results/MPE/speaker_listener_multiple/mappo/simple_speaker_listener/wandb/run-20250401_111625-sotjeafr/files" \
    --use_wandb False --share_policy --wheel_noise ${noise}
done
