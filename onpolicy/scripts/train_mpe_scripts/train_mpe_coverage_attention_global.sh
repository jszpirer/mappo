#!/bin/sh
env="MPE"
scenario="simple_coverage_attention" 
num_landmarks=0
num_agents=5
nb_additional_data=2
noise=0
algo="rmappo" #"mappo" "ippo"
exp="coverage_attention_global"
seed_max=5
project="simple_coverage_5agents_test"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
# for seed in `seq ${seed_max}`;
for seed in 1
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python -m cProfile -o output.prof ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 125 --num_env_steps 32000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-5 --critic_lr 7e-4 --user_name "jeanne-szpirer-universit-libre-de-bruxelles" --project_name ${project} \
    --nb_additional_data ${nb_additional_data} --wheel_noise ${noise} --attention_actor \
    --save_interval 1000000 --use_directions --d_model 32 --attention_critic
done
