#!/bin/bash
# Submission script for Lyra
#SBATCH --time=5-00:00:00 # days-hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres="gpu:1"
#SBATCH --mem=64G
#SBATCH --partition=batch
#
#SBATCH --output=out/%j_%x.out
#
echo "Loading modules"
module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load tensorboardX
module load SciPy-bundle
module load OpenAI-Gym
module load imageio
module load Seaborn
module load PyTorch
module load PyTorch-bundle
module load absl-py
module load wandb
module load Python-bundle-PyPI
echo "Modules loaded"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

export ScratchDir="${TMPDIR}/spread/"
mkdir -p ${ScratchDir}
cp -r ../../../../mappo/ ${ScratchDir}

cd ${ScratchDir}mappo/onpolicy/scripts/train_mpe_scripts/

export PYTHONPATH="${PYTHONPATH}:${ScratchDir}mappo/"

python3 -m venv --system-site-packages ${ScratchDir}env-localscratch/
source ${ScratchDir}env-localscratch/bin/activate
pip install -r /home/users/j/s/jszpirer/requirements.txt
export WANDB_API_KEY=f4f7b2b0294b94174927e70d87e96995ab195c62
wandb login

srun train_mpe_spread_cnn.sh

mkdir /home/users/j/s/jszpirer/%j_%x/

scp -r ${ScratchDir}mappo/onpolicy/scripts/results/MPE/simple_spread_cnn_sparse_3.85agentslandmarks/rmappo/simple_spread_cnn_sparse77/wandb/ /home/users/j/s/jszpirer/%j_%x/


