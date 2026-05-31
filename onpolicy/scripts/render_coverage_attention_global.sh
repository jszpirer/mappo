#!/bin/sh
env="MPE"
scenario="simple_coverage_cnn_sparse_localj_attention_global"
num_landmarks=0
num_agents=10
nb_additional_data=2
algo="rmappo"
exp="local_coverage_attention_newglobal_velocities"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    random_seed=$(date +%s%N | cut -b10-19)
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${random_seed} \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 125 --render_episodes 1 \
    --model_dir "/home/thales/jszpirer/sugar/mappo/onpolicy/scripts/results/coverage10agentsattentionglobal_seed2/files" \
    --use_wandb False --nb_additional_data ${nb_additional_data} --attention_actor \
    --omniscient_critic --velocities_critic --attention_critic --d_model 32
done
