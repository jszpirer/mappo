#!/bin/bash
# Submission script for Lyra
#SBATCH --time=2-00:00:00 # days-hh:mm:ss
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
module list
echo "Modules loaded"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

export CUDA_VISIBLE_DEVICES=1

echo "Localscratch dir"
echo $TMPDIR

export ScratchDir="${TMPDIR}/coverage/"
mkdir -p ${ScratchDir}
cp -r ../../../../mappo/ ${ScratchDir}
cp -r ../../../../env-mappo/ ${ScratchDir}

cd ${ScratchDir}mappo/onpolicy/scripts/train_mpe_scripts/

# export PYTHONPATH="/home/users/j/s/jszpirer/mappo/"
export PYTHONPATH="${ScratchDir}mappo/"

source ${ScratchDir}env-mappo/bin/activate

srun train_mpe_coverage_cnn.sh

mkdir /home/users/j/s/jszpirer/coverage5agentslocal/seed8

scp -r ${ScratchDir}mappo/onpolicy/scripts/results/MPE/simple_coverage_cnn_sparse_localj/rmappo/simple_coverage_cnn_sparse77_local/wandb/ /home/users/j/s/jszpirer/coverage5agents/seed8

